#!/bin/bash
cd $CODE_ROOT/src/kernels
bash build.sh
if [ ! -d "dist" ]; then
    echo "The directory of mie_ops wheel package 'dist' not found. Skipping wheel installation."
else
    echo "Start to install mie_ops found in the directory of mie_ops wheel package 'dist'."
    find dist -name "mie_ops*.whl" -exec pip install --force-reinstall --target $OUTPUT_DIR/lib {} \;
fi
cd -