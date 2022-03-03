files=$(
    grep -L "Copyright 2022 The Symanto Research Team Authors" $( find -name "*.py" )
)

for file in $files; do

    echo $file

    mv $file $file.bak
    cat dev-tools/preamble.txt $file.bak > $file
    rm $file.bak
done