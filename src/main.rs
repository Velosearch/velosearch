use clap::Parser;
use velosearch::index::PostingHandler;
use velosearch::{Result, FastArgs};
// use jemallocator::Jemalloc;
use tracing::{info, Level};

// #[global_allocator]
// static GLOBAL: Jemalloc = Jemalloc;

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();
    info!("main execution");
    let args = FastArgs::parse();

    let runtime = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    let mut handle = PostingHandler::new(args.base,
        args.path,
        args.partition_num.unwrap_or(1),
        args.batch_size.unwrap_or(512),
        args.dump_path
    );
    
    let res = runtime.block_on(async {handle.execute().await.unwrap() });
    println!("{:}", res);
    Ok(())
}
