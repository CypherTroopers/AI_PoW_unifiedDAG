~~~
./aidag-gen \
  --out ./aidag/colossusx-aidag-v1-128g.bin \
  --size-gb 128 \
  --model ./models/cypheriumai-light-v1-alpha.gguf \
  --model-offset-gb 32 \
  --seed "colossusx-ai-dag-v1" \
  --workers 12 \
  --chunk-mb 64 \
  --force
~~~
