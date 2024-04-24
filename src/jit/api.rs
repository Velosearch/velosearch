//! Constructing a function AST at runtime

use std::{collections::{HashMap, VecDeque}, sync::Arc, fmt::{Display, Formatter}};

use parking_lot::Mutex;

use crate::utils::{Result, FastErr};
use super::{ast::{JITType, Stmt, Expr, BOOL, TypedLit, Literal, BinaryExpr, NIL}, jit::JIT};

/// External Function signature
struct ExternFuncSignature {
    name: String,
    /// ptr to the func
    code: *const u8,
    params: Vec<JITType>,
    returns: Option<JITType>,
}

#[derive(Clone, Debug)]
/// A function consisting of AST nodes that JIT can compile.
pub struct GeneratedFunction {
    pub(crate) name: String,
    pub(crate) params: Vec<(String, JITType)>,
    pub(crate) body: Vec<Stmt>,
    pub(crate) ret: Option<(String, JITType)>,
}

#[derive(Default)]
/// State pf Assembler, keep tracking of generated function names
/// and registered external functions.
pub struct  AssemblerState {
    name_next_id: HashMap<String, u8>,
    extern_funcs: HashMap<String, ExternFuncSignature>,
}

impl AssemblerState {
    /// Create a fresh function name with prefix `name`.
    pub fn fresh_name(&mut self, name: impl Into<String>) -> String {
        let name = name.into();
        if !self.name_next_id.contains_key(&name) {
            self.name_next_id.insert(name.clone(), 0);
        }

        let id = self.name_next_id.get_mut(&name).unwrap();
        let name = format!("{}_{}", &name, id);
        *id += 1;
        name
    }
}

/// The very first step for constructing a function at runtime
pub struct Assembler {
    state: Arc<Mutex<AssemblerState>>,
}

impl Default for Assembler {
    fn default() -> Self {
        Self {
            state: Arc::new(Default::default()),
        }
    }
}

impl Assembler {
    /// Register and external Rust function to make it accessible by runtime generated functions.
    /// Parameters and return types are used to impose type safety while constructing an AST.
    pub fn register_extern_fn(
        &self,
        name: impl Into<String>,
        ptr: *const u8,
        params: Vec<JITType>,
        returns: Option<JITType>,
    ) -> Result<()> {
        let extern_funcs = &mut self.state.lock().extern_funcs;
        let fn_name = name.into();
        let old = extern_funcs.insert(
            fn_name.clone(),
            ExternFuncSignature {
                name: fn_name,
                code: ptr,
                params,
                returns
            },
        );

        match old {
            None => Ok(()),
            Some(old) => Err(FastErr::InternalErr(format!("Extern function {} already exists", old.name))),
        }
    }

    /// Create a new FunctionBuilder with "name" prefix
    pub fn new_func_builder(&self, name: impl Into<String>) -> FunctionBuilder {
        let name = self.state.lock().fresh_name(name);
        FunctionBuilder::new(name, self.state.clone())
    }

    /// Create JIT env which we could compile the AST of constructed function
    /// into runable code
    pub fn create_jit(&self) -> JIT {
        let symbols = self
            .state
            .lock()
            .extern_funcs
            .values()
            .map(|s| (s.name.clone(), s.code))
            .collect::<Vec<_>>();
        JIT::new(symbols)
    }
}

/// Function builder API. Stores the state while
/// we are constructing an AST for a function
pub struct  FunctionBuilder {
    name: String,
    params: Vec<(String, JITType)>,
    ret: Option<(String, JITType)>,
    fields: VecDeque<HashMap<String, JITType>>,
    assembler_state: Arc<Mutex<AssemblerState>>,
}

impl FunctionBuilder {
    fn new(name: impl Into<String>, assembler_state: Arc<Mutex<AssemblerState>>) -> Self {
        let mut fields = VecDeque::new();
        fields.push_back(HashMap::new());
        Self {
            name: name.into(),
            params: Vec::new(),
            ret: None,
            fields,
            assembler_state,
        }
    }

    /// Add one more parameter to the function.
    #[must_use]
    pub fn param(mut self, name: impl Into<String>, ty: JITType) -> Self {
        let name = name.into();
        assert!(!self.fields.back().unwrap().contains_key(&name));
        self.params.push((name.clone(), ty));
        self.fields.back_mut().unwrap().insert(name, ty);
        self
    }

    /// Set return type for the function. Functions are of `void` type by default if
    /// you do not set the return type.
    #[must_use]
    pub fn ret(mut self, name: impl Into<String>, ty: JITType) -> Self {
        let name = name.into();
        assert!(!self.fields.back().unwrap().contains_key(&name));
        self.ret = Some((name.clone(), ty));
        self.fields.back_mut().unwrap().insert(name, ty);
        self
    }

    /// Enter the function body at start the building.
    pub fn enter_block(&mut self) -> CodeBlock {
        self.fields.push_back(HashMap::new());
        CodeBlock {
            fields: &mut self.fields,
            state: &self.assembler_state,
            stmts: vec![],
            while_state: None,
            if_state: None,
            fn_state: Some(GeneratedFunction {
                name: self.name.clone(),
                params: self.params.clone(),
                body: vec![],
                ret: self.ret.clone(),
            }),
        }
    }
}

/// Keep `while` condition expr as we are constructing while loop body.
struct WhileState {
    condition: Expr,
}

/// Keep `if-then-else` state, including condition expr, the already built
/// then statements (if we are during building the else block).
struct IfElseState {
    condition: Expr,
    then_stmts: Vec<Stmt>,
    in_then: bool,
}

impl IfElseState {
    /// Move the all current statements in the `then` block and move to `else` block
    fn enter_else(&mut self, then_stmts: Vec<Stmt>) {
        self.then_stmts = then_stmts;
        self.in_then = false;
    }
}

/// Code block consists of statements and acts as anonymous namespace scope for items and variable
/// declarations.
pub struct CodeBlock<'a> {
    /// A stack that containing all defined variables so far. The variables defined
    /// in the current block are at the top stack frame.
    /// Fields provides a shadow semantics of the same name in outsider block, and are
    /// used to gurantee type safety while constructing AST.
    fields: &'a mut VecDeque<HashMap<String, JITType>>,
    /// The state of Assembler, used for type checking function calls.
    state: &'a Arc<Mutex<AssemblerState>>,
    /// Holding all statements for the current code block.
    stmts: Vec<Stmt>,
    while_state: Option<WhileState>,
    if_state: Option<IfElseState>,
    // Keep track of function params and return types, only valid for function main block.
    fn_state: Option<GeneratedFunction>,
}

impl<'a> CodeBlock<'a> {
    pub fn build(&mut self) -> GeneratedFunction {
        assert!(
            self.fn_state.is_some(),
            "Can only call build on outermost function block"
        );
        let mut gen = self.fn_state.take().unwrap();
        gen.body = self.stmts.drain(..).collect::<Vec<_>>();
        gen
    }

    /// Leave the current block and returns the statements constructed.
    fn leave(&mut self) -> Result<Stmt> {
        self.fields.pop_back();
        if let Some(ref mut while_state) = self.while_state {
            let WhileState { condition } = while_state;
            let stmts = self.stmts.drain(..).collect::<Vec<_>>();
            return Ok(Stmt::WhileLoop(Box::new(condition.clone()), stmts));
        }

        if let Some(ref mut if_state) = self.if_state {
            let IfElseState {
                condition,
                then_stmts,
                in_then,
            } = if_state;
            return if *in_then {
                assert!(then_stmts.is_empty());
                let stmts = self.stmts.drain(..).collect::<Vec<_>>();
                Ok(Stmt::IfElse(Box::new(condition.clone()), stmts, Vec::new()))
            } else {
                assert!(!then_stmts.is_empty());
                let then_stmts = then_stmts.drain(..).collect::<Vec<_>>();
                let else_stmts = self.stmts.drain(..).collect::<Vec<_>>();
                Ok(Stmt::IfElse(
                    Box::new(condition.clone()),
                    then_stmts,
                    else_stmts,
                ))
            }
        }
        unreachable!()
    }

    /// Enter else block. Try [if block] first which is much easier to use.
    fn enter_else(&mut self) {
        self.fields.pop_back();
        self.fields.push_back(HashMap::new());
        assert!(self.if_state.is_some() && self.if_state.as_ref().unwrap().in_then);
        let new_then = self.stmts.drain(..).collect::<Vec<_>>();
        if let Some(s) = self.if_state.iter_mut().next() {
            s.enter_else(new_then)
        }
    }

    /// Declare variable `name` of a type
    pub fn declare(&mut self, name: impl Into<String>, ty: JITType) -> Result<()> {
        let name = name.into();
        let typ = self.fields.back().unwrap().get(&name);
        match typ {
            Some(typ) => Err(FastErr::InternalErr(
                format!(
                    "Variable {} of {} already exits in the current scope",
                    name,
                    typ,
                )
            )),
            None => {
                self.fields.back_mut().unwrap().insert(name.clone(), ty);
                self.stmts.push(Stmt::Declare(name, ty));
                Ok(())
            }
        }
    }

    fn find_type(&self, name: impl Into<String>) -> Option<JITType> {
        let name = name.into();
        for scope in self.fields.iter().rev() {
            let typ = scope.get(&name);
            if let Some(typ) = typ {
                return Some(*typ);
            }
        }
        None
    }

     /// Assignment statement. Assign a expression value to a variable.
     pub fn assign(&mut self, name: impl Into<String>, expr: Expr) -> Result<()> {
        let name = name.into();
        let typ = self.find_type(&name);
        match typ {
            Some(typ) => {
                if typ != expr.get_type() {
                    Err(FastErr::InternalErr(
                        format!(
                            "Variable {} of {} cannot be assigned to {}",
                            name,
                            typ,
                            expr.get_type()
                        )
                    ))
                } else {
                    self.stmts.push(Stmt::Assign(name, Box::new(expr)));
                    Ok(())
                }
            }
            None => Err(FastErr::InternalErr(format!("unknown identifier: {}", name))),
        }
    }

    /// Declare variable with initialization.
    pub fn declare_as(&mut self, name: impl Into<String>, expr: Expr) -> Result<()> {
        let name = name.into();
        let typ = self.fields.back().unwrap().get(&name);
        match typ {
            Some(typ) => {
                Err(FastErr::InternalErr(
                    format!(
                        "Variable {} of {} already exists in the current scope",
                        name,
                        typ
                    )
                ))
            }
            None => {
                self.fields
                    .back_mut()
                    .unwrap()
                    .insert(name.clone(), expr.get_type());
                self.stmts
                    .push(Stmt::Declare(name.clone(), expr.get_type()));
                self.stmts.push(Stmt::Assign(name, Box::new(expr)));
                Ok(())
            }
        }
    }

    /// Call external function for side effect only.
    pub fn call_stmt(&mut self, name: impl Into<String>, args: Vec<Expr>) -> Result<()> {
        self.stmts.push(Stmt::Call(name.into(), args));
        Ok(())
    }

    /// Enter `while` loop block. Try [while_block] first which is much easier to use.
    fn while_loop(&mut self, cond: Expr) -> Result<CodeBlock> {
        if cond.get_type() != BOOL {
            Err(FastErr::InternalErr(format!("while condition must be bool")))
        } else {
            self.fields.push_back(HashMap::new());
            Ok(CodeBlock {
                fields: self.fields,
                state: self.state,
                stmts: vec![],
                while_state: Some(WhileState { condition: cond }),
                if_state: None,
                fn_state: None,
            })
        }
    }

    /// Enter `if-then-else`'s then block. Try [if_block] first which is much easier to use.
    fn if_else(&mut self, cond: Expr) -> Result<CodeBlock> {
        if cond.get_type() != BOOL {
            Err(FastErr::InternalErr(format!("if condition must be bool")))
        } else {
            self.fields.push_back(HashMap::new());
            Ok(CodeBlock {
                fields: self.fields,
                state: self.state,
                stmts: vec![],
                while_state: None,
                if_state: Some(IfElseState {
                    condition: cond,
                    then_stmts: vec![],
                    in_then: true,
                }),
                fn_state: None,
            })
        }
    } 

    /// Construct a `if-then-else` block with each part provided.
    ///
    /// E.g. if n == 0 { r = 0 } else { r = 1} could be write as:
    /// x.if_block(
    ///    |cond| cond.eq(cond.id("n")?, cond.lit_i(0)),
    ///    |t| {
    ///        t.assign("r", t.lit_i(0))?;
    ///        Ok(())
    ///    },
    ///    |e| t.assign("r", t.lit_i(1))?;
    ///       Ok(())
    ///    },
    /// )?;
    pub fn if_block<C, T, E>(
        &mut self,
        mut cond: C,
        mut then_blk: T,
        mut else_blk: E,
    ) -> Result<()>
    where
        C: FnMut(&mut CodeBlock) -> Result<Expr>,
        T: FnMut(&mut CodeBlock) -> Result<()>,
        E: FnMut(&mut CodeBlock) -> Result<()>,
    {
        let cond = cond(self)?;
        let mut body = self.if_else(cond)?;
        then_blk(&mut body)?;
        body.enter_else();
        else_blk(&mut body)?;
        let if_else = body.leave()?;
        self.stmts.push(if_else);
        Ok(())
    }

    /// Construct a `while` block with each part provided.
    ///
    /// E.g. while n != 0 { n = n - 1;} could be write as:
    /// x.while_block(
    ///    |cond| cond.ne(cond.id("n")?, cond.lit_i(0)),
    ///    |w| {
    ///        w.assign("n", w.sub(w.id("n")?, w.lit_i(1))?)?;
    ///        Ok(())
    ///    },
    /// )?;
    pub fn while_block<C, B>(&mut self, mut cond: C, mut body_blk: B) -> Result<()>
    where
        C: FnMut(&mut CodeBlock) -> Result<Expr>,
        B: FnMut(&mut CodeBlock) -> Result<()>,
    {
        let cond = cond(self)?;
        let mut body = self.while_loop(cond)?;
        body_blk(&mut body)?;
        let while_stmt = body.leave()?;
        self.stmts.push(while_stmt);
        Ok(())
    }

    /// Create a literal `val` of `ty` type.
    pub fn lit(&self, val: impl Into<String>, ty: JITType) -> Expr {
        Expr::Literal(Literal::Parsing(val.into(), ty))
    }

    /// Shorthand to create u8 literal
    pub fn lit_u8(&self, val: impl Into<u8>) -> Expr {
        Expr::Literal(Literal::Typed(TypedLit::U8(val.into())))
    }

    /// Shorthand to create u16 literal
    pub fn lit_u16(&self, val: impl Into<u16>) -> Expr {
        Expr::Literal(Literal::Typed(TypedLit::U16(val.into())))
    }

    /// Shorthand to create boolean literal
    pub fn lit_b(&self, val: impl Into<u8>) -> Expr {
        Expr::Literal(Literal::Typed(TypedLit::Bool(val.into())))
    }

    /// Shorthand to create i64 literal
    pub fn lit_i64(&self, val: impl Into<i64>) -> Expr {
        Expr::Literal(Literal::Typed(TypedLit::I64(val.into())))
    }

    /// Create a reference to an already defined variable.
    pub fn id(&self, name: impl Into<String>) -> Result<Expr> {
        let name = name.into();
        match self.find_type(&name) {
            None => Err(FastErr::InternalErr(format!("unknown identifier: {}", name))),
            Some(typ) => Ok(Expr::Identifier(name, typ)),
        }
    }

     /// Binary comparison expression: lhs == rhs
     pub fn eq(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot compare {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::Eq(Box::new(lhs), Box::new(rhs))))
        }
    } 

    /// Binary comparison expression: lhs != rhs
    pub fn ne(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot compare {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::Ne(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Binary comparison expression: lhs < rhs
    pub fn lt(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot compare {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::Lt(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Binary comparison expression: lhs <= rhs
    pub fn le(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot compare {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::Le(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Binary comparison expression: lhs > rhs
    pub fn gt(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot compare {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::Gt(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Binary comparison expression: lhs >= rhs
    pub fn ge(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot compare {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::Ge(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Binary arithmetic expression: lhs + rhs
    pub fn add(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot add {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::Add(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Binary arithmetic expression: lhs - rhs
    pub fn sub(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot subtract {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::Sub(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Binary arithmetic expression: lhs * rhs
    pub fn mul(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot multiply {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::Mul(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Binary arithmetic expression: lhs / rhs
    pub fn div(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot divide {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::Div(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Binary arithmetic expression: lhs & rhs
    pub fn bitiwise_and(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot bitwise and {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::BitwiseAnd(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Binary arithmetic expression: lsh | rhs
    pub fn bitwise_or(&self, lhs: Expr, rhs: Expr) -> Result<Expr> {
        if lhs.get_type() != rhs.get_type() {
            internal_err!("cannot bitwise or {} and {}", lhs.get_type(), rhs.get_type())
        } else {
            Ok(Expr::Binary(BinaryExpr::BitwiseOr(Box::new(lhs), Box::new(rhs))))
        }
    }

    /// Call external function `name` with parameters
    pub fn call(&self, name: impl Into<String>, params: Vec<Expr>) -> Result<Expr> {
        let fn_name = name.into();
        if let Some(func) = self.state.lock().extern_funcs.get(&fn_name) {
            for ((i, t1), t2) in params.iter().enumerate().zip(func.params.iter()) {
                if t1.get_type() != *t2 {
                    return internal_err!(
                        "Func {} need {} as arg{}, get {}",
                        &fn_name,
                        t2,
                        i,
                        t1.get_type()
                    );
                }
            }
            Ok(Expr::Call(fn_name, params, func.returns.unwrap_or(NIL)))
        } else {
            internal_err!("No func with the name {} exist", fn_name)
        }
    }

    /// Return the value pointed to by the ptr stored in `ptr`
    pub fn load(&self, ptr: Expr, ty: JITType) -> Result<Expr> {
        Ok(Expr::Load(Box::new(ptr), ty))
    }

    /// Store the value in `value` to the address in `ptr`
    pub fn store(&mut self, value: Expr, ptr: Expr) -> Result<()> {
        self.stmts.push(Stmt::Store(Box::new(value), Box::new(ptr)));
        Ok(())
    }
}

impl Display for GeneratedFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "fn {}(", self.name)?;
        for (i, (name, ty)) in self.params.iter().enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{name}: {ty}")?;
        }
        write!(f, ") -> ")?;
        if let Some((name, ty)) = &self.ret {
            write!(f, "{name}: {ty}")?;
        } else {
            write!(f, "()")?;
        }
        writeln!(f, " {{")?;
        for stmt in &self.body {
            stmt.fmt_ident(4, f)?;
        }
        write!(f, "}}")
    }
}

macro_rules! internal_err {
    ($($arg:tt)*) => {
        Err(FastErr::InternalErr(format!($($arg)*)))
    };
}

use internal_err;