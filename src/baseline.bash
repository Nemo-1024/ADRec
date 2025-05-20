
datasets=('yelp')
#datasets=('baby' 'beauty' 'ml-100k' 'sports' 'toys' 'yelp')
#models=("adrec" "diffurec"  "dreamrec")
models=('adrec')
device='cuda:0'

for j in "${models[@]}"; do
    for i in "${datasets[@]}"; do
      echo "Running experiment: ${i}"
      python main.py --dataset "${i}" --model "${j}" --description "_" --device "${device}"
    done
done
wait
echo "All experiments are done!"