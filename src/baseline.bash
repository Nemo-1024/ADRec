# 定义数据集数组
datasets=('yelp')
#datasets=('baby' 'beauty' 'ml-100k' 'sports' 'toys' 'yelp')
#models=("lightsans" "bert4rec" "core" "eulerformer" "fearec" "gru4rec")
#models=("diffurec" "adrec" "pretrain" "dreamrec")
models=('adrec')
device='cuda:1'
# 遍历数据集数组，运行每个实验
for j in "${models[@]}"; do
    for i in "${datasets[@]}"; do
      echo "Running experiment: ${i}"
      python main.py --dataset "${i}" --model "${j}" --description "_" --device "${device}"
    done
done
wait
echo "All experiments are done!"