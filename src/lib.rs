
#![feature(portable_simd)]
#![feature(stdsimd)]
#![feature(is_sorted)]
#![feature(sync_unsafe_cell)]
#![feature(slice_flatten)]
#![feature(ptr_metadata)]
#![feature(once_cell)]
#![feature(stmt_expr_attributes)]

extern crate datafusion;

pub mod utils;
pub mod index;
pub mod optimizer;
pub mod datasources;
pub mod jit;
pub mod query;
pub mod context;
pub mod batch;
pub mod physical_expr;
use datafusion::prelude::*;
// use jemallocator::Jemalloc;
use tokio::sync::Mutex;
pub use utils::Result;
pub use context::BooleanContext;
pub use optimizer::{BooleanPhysicalPlanner, MinOperationRange, PartitionPredicateReorder, RewriteBooleanPredicate, PrimitivesCombination};
pub use physical_expr::ShortCircuit;
use clap::Parser;
use tantivy::tokenizer::{TextAnalyzer, SimpleTokenizer, RemoveLongFilter, LowerCaser};
use lazy_static::lazy_static;
use peg;

// #[global_allocator]
// static GLOBAL: Jemalloc = Jemalloc;
lazy_static!{
    pub static ref TOKENIZER: TextAnalyzer = TextAnalyzer::from(SimpleTokenizer)
    .filter(RemoveLongFilter::limit(60))
    .filter(LowerCaser);
}

fn col_tokenized(token: &str) -> Expr {
    let mut process = TOKENIZER.token_stream(&token);
    let mut exprs = Vec::new();
    process.process(&mut |token| {
        exprs.push(Expr::Column(token.text.clone().into()));
    });
    exprs.into_iter()
    .reduce(|l, r| {
        boolean_and(l, r)
    })
    .expect(&format!("token: {:}", token))
}

peg::parser!{
    pub grammar parser() for str {
        pub rule boolean() -> Expr = precedence!{
            x:(@) _ "AND" _ y:@ { boolean_and(x, y)}
            x:(@) _ "OR" _ y:@ { boolean_or(x, y) }
            --
            n:term() { n }
            "(" e:boolean()")" { e }
        }

        rule term() -> Expr 
            = e:$(['a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '\'' | '.' | '-']+) { col_tokenized(e) }

        rule _() =  quiet!{[' ' | '\t']*}
    }
}

peg::parser!{
    pub grammar boolean_parser() for str {
        pub rule boolean() -> Expr = precedence!{
            x:and() _ y:(@) { boolean_and(x, y) }
            x:or() _ y:(@) { boolean_or(x, y)}
            x:and() { x }
            x:or() { x }
            --
            "(" e:boolean() ")" { e }
            "+(" e:boolean() ")" { e }
        }
        
        rule or() -> Expr
            = l:term() { l }

        rule and() -> Expr
            = "+" e:term() {e}

        rule term() -> Expr 
            = e:$(['a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '.' ]+) { col_tokenized(e) }

        rule _() =  quiet!{[' ' | '\t']*}
    }
}

const JIT_MAX_NODES: usize = 0;

lazy_static!{
    pub static ref STEP_LEN: Mutex<usize> = {
        Mutex::new(1)
    };
}

lazy_static!{
    pub static ref CONCURRENCY: Mutex<usize> = {
        Mutex::new(1)
    };
}

#[derive(Parser, Debug)]
pub struct FastArgs {
    /// file path
    pub path: Vec<String>,

    #[arg(short, long)]
    pub partition_num: Option<usize>,

    #[arg(short, long)]
    pub batch_size: Option<u32>,

    #[arg(long)]
    pub base: String,

    #[arg(long, short)]
    pub dump_path: Option<String>,
}