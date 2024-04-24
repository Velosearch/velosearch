use cranelift::codegen::ir;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug)]
/// Statement
pub enum Stmt {
    /// if-then-else
    IfElse(Box<Expr>, Vec<Stmt>, Vec<Stmt>),
    /// while
    WhileLoop(Box<Expr>, Vec<Stmt>),
    /// assignment
    Assign(String, Box<Expr>),
    /// call function for side effect
    Call(String, Vec<Expr>),
    /// declare a new variable of type
    Declare(String, JITType),
    /// store value (the first expr) tp an address (the second expr)
    Store(Box<Expr>, Box<Expr>),

}

impl Stmt {
    /// print the statement with indentation
    pub fn fmt_ident(&self, ident: usize, f: &mut Formatter) -> std::fmt::Result {
        let mut ident_str = String::new();
        for _ in 0..ident {
            ident_str.push(' ');
        }
        match self {
            Stmt::IfElse(cond, then_stmts, else_stmts) => {
                writeln!(f, "{ident_str}if {cond} {{")?;
                for stmt in then_stmts {
                    stmt.fmt_ident(ident + 4, f)?;
                }
                writeln!(f, "{ident_str}}} else {{")?;
                for stmt in else_stmts {
                    stmt.fmt_ident(ident + 4, f)?;
                }
                writeln!(f, "{ident_str}}}")
            }
            Stmt::WhileLoop(cond, stmts) => {
                writeln!(f, "{ident_str}while {cond} {{")?;
                for stmt in stmts {
                    stmt.fmt_ident(ident + 4, f)?;
                }
                writeln!(f, "{ident_str}}}")
            }
            Stmt::Assign(name, expr) => {
                writeln!(f, "{ident_str}{name} = {expr};")
            }
            Stmt::Call(name, args) => {
                writeln!(
                    f,
                    "{}{}({});",
                    ident_str,
                    name,
                    args.iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Stmt::Declare(name, ty) => {
                writeln!(f, "{ident_str}let {name}: {ty};")
            }
            Stmt::Store(value, ptr) => {
                writeln!(f, "{ident_str}*({ptr}) = {value}")
            }
        }
    }
}

impl Display for Stmt {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.fmt_ident(0, f)?;
        Ok(())
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
/// Shorthand typed literals
pub enum TypedLit {
    U8(u8),
    U16(u16),
    I64(i64),
    Bool(u8),
}

impl TypedLit {
    fn get_type(&self) -> JITType {
        match self {
            TypedLit::U8(_) => U8,
            TypedLit::U16(_) => U16,
            TypedLit::I64(_) => I64,
            TypedLit::Bool(_) => BOOL,
        }
    }
}

impl Display for TypedLit {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TypedLit::Bool(b) => write!(f, "{b}"),
            TypedLit::U8(u) => write!(f, "{u}"),
            TypedLit::U16(u) => write!(f, "{u}"),
            TypedLit::I64(i) => write!(f, "{i}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Expression
pub enum Expr {
    /// Literal
    Literal(Literal),
    /// Variable
    Identifier(String, JITType),
    /// Binary Expression
    Binary(BinaryExpr),
    /// call function expression
    Call(String, Vec<Expr>, JITType),
    /// Load a value from pointer
    Load(Box<Expr>, JITType),
    /// Boolean Expression
    BooleanExpr(BooleanExpr),
    /// Boolean Expression V2
    Boolean(Boolean),
}

impl Expr {
    pub fn get_type(&self) -> JITType {
        match self {
            Expr::Literal(lit) => lit.get_type(),
            Expr::Identifier(_, ty) => *ty,
            Expr::Binary(bin) => bin.get_type(),
            Expr::Call(_, _, ty) => *ty,
            Expr::Load(_, ty) => *ty,
            Expr::BooleanExpr(b) => b.get_type(),
            Expr::Boolean(b) => b.get_type(),
        }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Literal(l) => write!(f, "{l}"),
            Expr::Identifier(name, _) => write!(f, "{name}"),
            Expr::Binary(be) => write!(f, "{be}"),
            Expr::BooleanExpr(b) => write!(f, "{b}"),
            Expr::Boolean(b) => write!(f, "{b}"),
            Expr::Call(name, exprs, _) => {
                write!(
                    f,
                    "{}({})",
                    name,
                    exprs
                        .iter()
                        .map(|e| e.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Expr::Load(ptr, _) => write!(f, "*({ptr})",),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
/// Type to be used in JIT
pub struct JITType {
    /// The cranelift type
    pub native: ir::Type,
    /// code
    pub code: u8,
}

impl std::fmt::Display for JITType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::fmt::Debug for JITType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.code {
            0 => write!(f, "nil"),
            0x70 => write!(f, "bool"),
            0x76 => write!(f, "i8"),
            0x77 => write!(f, "i16"),
            0x78 => write!(f, "i32"),
            0x79 => write!(f, "i64"),
            0x7b => write!(f, "f32"),
            0x7c => write!(f, "f64"),
            0x7e => write!(f, "small_ptr"),
            0x7f => write!(f, "ptr"),
            _ => write!(f, "unknown"),
        }
    }
}

impl From<&str> for JITType {
    fn from(x: &str) -> Self {
        match x {
            "bool" => BOOL,
            "u8" => U8,
            "U16" => U16,
            "small_ptr" => R32,
            "ptr" => R64,
            _ => panic!("unknown type: {x}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Literal
pub enum Literal {
    /// Parsable literal with type
    Parsing(String, JITType),
    /// Shorthand literals of common types
    Typed(TypedLit),
}

impl Literal {
    fn get_type(&self) -> JITType {
        match self {
            Literal::Parsing(_, ty) => *ty,
            Literal::Typed(tl) => tl.get_type(),
        }
    }
}

impl Display for Literal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Literal::Parsing(str, _) => write!(f, "{str}"),
            Literal::Typed(tl) => write!(f, "{tl}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Binary Expression
pub enum BinaryExpr {
    /// ==
    Eq(Box<Expr>, Box<Expr>),
    /// !=
    Ne(Box<Expr>, Box<Expr>),
    /// <
    Lt(Box<Expr>, Box<Expr>),
    /// <=
    Le(Box<Expr>, Box<Expr>),
    /// >
    Gt(Box<Expr>, Box<Expr>),
    /// >=
    Ge(Box<Expr>, Box<Expr>),
    /// add
    Add(Box<Expr>, Box<Expr>),
    /// subtract
    Sub(Box<Expr>, Box<Expr>),
    /// multiply
    Mul(Box<Expr>, Box<Expr>),
    /// divide
    Div(Box<Expr>, Box<Expr>),
    /// Bitwise and
    BitwiseAnd(Box<Expr>, Box<Expr>),
    /// Bitwise or
    BitwiseOr(Box<Expr>, Box<Expr>),
}

impl BinaryExpr {
    fn get_type(&self) -> JITType {
        match self {
            BinaryExpr::Eq(_, _) => BOOL,
            BinaryExpr::Ne(_, _) => BOOL,
            BinaryExpr::Lt(_, _) => BOOL,
            BinaryExpr::Le(_, _) => BOOL,
            BinaryExpr::Gt(_, _) => BOOL,
            BinaryExpr::Ge(_, _) => BOOL,
            BinaryExpr::Add(lhs, _) => lhs.get_type(),
            BinaryExpr::Sub(lhs, _) => lhs.get_type(),
            BinaryExpr::Mul(lhs, _) => lhs.get_type(),
            BinaryExpr::Div(lhs, _) => lhs.get_type(),
            BinaryExpr::BitwiseAnd(lhs, _) => lhs.get_type(),
            BinaryExpr::BitwiseOr(lhs, _) => lhs.get_type(),
            
        }
    }
}

impl Display for BinaryExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryExpr::Eq(lhs, rhs) => write!(f, "{lhs} == {rhs}"),
            BinaryExpr::Ne(lhs, rhs) => write!(f, "{lhs} != {rhs}"),
            BinaryExpr::Lt(lhs, rhs) => write!(f, "{lhs} < {rhs}"),
            BinaryExpr::Le(lhs, rhs) => write!(f, "{lhs} <= {rhs}"),
            BinaryExpr::Gt(lhs, rhs) => write!(f, "{lhs} > {rhs}"),
            BinaryExpr::Ge(lhs, rhs) => write!(f, "{lhs} >= {rhs}"),
            BinaryExpr::Add(lhs, rhs) => write!(f, "{lhs} + {rhs}"),
            BinaryExpr::Sub(lhs, rhs) => write!(f, "{lhs} - {rhs}"),
            BinaryExpr::Mul(lhs, rhs) => write!(f, "{lhs} * {rhs}"),
            BinaryExpr::Div(lhs, rhs) => write!(f, "{lhs} / {rhs}"),
            BinaryExpr::BitwiseAnd(lhs, rhs) => write!(f, "{lhs} & {rhs}"),
            BinaryExpr::BitwiseOr(lhs, rhs) => write!(f, "{lhs} | {rhs}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Predicate {
    And { args: Vec<Predicate> },
    Or { args: Vec<Predicate> },
    Leaf { idx: usize },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Boolean {
    pub predicate: Predicate,
    pub start_idx: usize,
}

impl Boolean {
    fn get_type(&self) -> JITType {
        I64
    }
}

impl Display for Boolean {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "eval({:?})", self.predicate)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BooleanExpr {
    pub(crate) cnf: Vec<i64>,
}

impl BooleanExpr {
    fn get_type(&self) -> JITType {
        U8
    }
}

impl Display for BooleanExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "eval({:?})", self.cnf)
    }
}

/// null type as placeholder
pub const NIL: JITType = JITType {
    native: ir::types::INVALID,
    code: 0,
};

/// bool, use u8 as bool
pub const BOOL: JITType = JITType {
    native: ir::types::I8,
    code: 0x76,
};

/// u8
pub const U8: JITType = JITType {
    native: ir::types::I8,
    code: 0x76
};

/// U16
pub const U16: JITType = JITType {
    native: ir::types::I16,
    code: 0x77,
};
/// integer of 8 bytes
pub const I64: JITType = JITType {
    native: ir::types::I64,
    code: 0x79,
};

pub const R32: JITType = JITType {
    native: ir::types::R32,
    code: 0x7e,
};
/// Pointer type of 64 bits
pub const R64: JITType = JITType {
    native: ir::types::R64,
    code: 0x7f,
};
pub const PTR_SIZE: usize = std::mem::size_of::<usize>();
/// The pointer type to use based on our currently target.
pub const PTR: JITType = if PTR_SIZE == 8 { R64 } else { R32 };