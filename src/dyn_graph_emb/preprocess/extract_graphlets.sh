#!/bin/bash

#INPUT_DIR=="~/dyn-graph-emb/"
FILE_PATH="/home/ninew/dyn-graph-emb/data/age_filter/age_30_65.txt" # list of files
FILE_DIR="/home/ninew/dyn-graph-emb/data/prep_w50_s5_aal_all" # directory of files
OUTPUT_DIR="/home/ninew/dyn-graph-emb/output/dgraphlets/age_30_65"
  # Limit to 5 parallel jobs
#~/code/dcount ~/dyn-graph-emb/data/prep_w50_s8_aal_batch1/CMU_a_0050642_func_preproc.csv 5 3 1

#function process_file {
#  bash "~/code/dcount" "$1" "5 3 1" "output path" "-v ./vector_files/graphlets_5_3.txt -g ./vector_files/orbits_5_3.txt -d ' '"
#}
#./dcount ../graphlet_adr/data/dppi/workplace/dgraphlet_format/network_6h.txt 4 5 1 ../graphlet_adr/data/dppi/workplace/output_6h/output -v ./vector_files/graphlets_4_5.txt -g ./vector_files/orbits_4_5.txt -d ' '

function process_file {
  local FILE=$1
  /home/ninew/code/dcount \
  "$FILE_DIR/${FILE}_func_preproc.csv" \
   5 3 1 \
   "$OUTPUT_DIR/$FILE" \
   -v /home/ninew/code/vector_files/graphlets_5_3.txt \
   -g /home/ninew/code/vector_files/orbits_5_3.txt \
   -d ","
}
#!/bin/bash

# Max number of parallel jobs
MAX_JOBS=1

# Counter for current number of parallel jobs
count=0

# Read each line in the file
while read -r filename; do
    # Print the filename being processed
    echo "===========Processing $filename...==========="

    process_file "$filename" &

    # Increment the counter
    ((count++))

    # Check if we need to wait
    if [ "$count" -ge "$MAX_JOBS" ]; then
        wait -n  # Wait for at least one process to finish
        ((count--))  # Decrement counter once a job finishes
    fi
done < $FILE_PATH

# Wait for all background jobs to finish
wait
