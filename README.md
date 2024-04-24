# VeloSearch
This is the source code of the submissioned paper at SIGMOD'25: Enabling Efficient In-Memory Full-Text Search with Vectorization on Compacted Columnar Format.

## Env requirement
Rust version: 1.68.0

OS: 20.04.1-Ubuntu

CPU: x86\_64 and supports AVX512, AVX, SSE4.0 and SSE SIMD instruction extensions.

# Setup
## Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s --default-toolchain=1.68.0
```

## Pull dependences
Pull the customized Apache Datafusion library.
```bash
git submodule update
```

## Compile
```bash
cargo build --release
```

# Usage
## Evalute batched search queries
Command: (the bin is generated after the rustc compilation)
```bash
./target/release/fastful-search
```

help info:
```
Usage: fastfull-search [OPTIONS] --handler <HANDLER> --base <BASE> [PATH]...

Arguments:
  [PATH]...  file path

Options:
      --handler <HANDLER>              [possible values: base, split-base, split-o1, load-data, boolean-query, posting-table, tantivy]
  -p, --partition-num <PARTITION_NUM>  
  -b, --batch-size <BATCH_SIZE>        
      --base <BASE>                    
  -d, --dump-path <DUMP_PATH>          
  -h, --help                           Print help
```

## Interactive full-text search mode:
Command: (the bin is generated after the compilation)
```bash
./target/release/do_query <idx_dir> <thread_num>
```