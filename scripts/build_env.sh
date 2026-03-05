#!/bin/bash
export thread_num=${MAX_COMPILE_CORE_NUM:-$(nproc)}
echo "Compilation thread number set to: $thread_num"
THIRD_PARTY_OUTPUT_DIR=$CODE_ROOT/third_party/output
CACHE_DIR=$CODE_ROOT/.cache
BUILD_DIR=$CODE_ROOT/build
OUTPUT_DIR=$CODE_ROOT/output
RELEASE_DIR=$CODE_ROOT/release
MINDIE_LLM_DIR=$CODE_ROOT/mindie_llm
MINDIE_LLM_LIB_DIR=$MINDIE_LLM_DIR/lib
LOG_PATH="/var/log/mindie_log/"
LOG_NAME="mindie_llm_install.log"
ARCH="aarch64"
if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
    ARCH="x86_64"
    echo "It is system of x86_64"
elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
    echo "It is system of aarch64"
else
    echo "It is not system of aarch64 or x86_64"
fi
COMPILE_OPTIONS=""
build_type="debug"
BUILD_OPTION_LIST="master debug release help dlt unittest 3rd clean"
BUILD_CONFIGURE_LIST=("--use_cxx11_abi=0" "--use_cxx11_abi=1" "--ini=version" "--ini=version_item")
export LANG=C.UTF-8
export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$PYTORCH_INSTALL_PATH/../torch.libs:$LD_LIBRARY_PATH
