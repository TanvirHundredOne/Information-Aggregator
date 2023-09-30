<h1> Info Aggregator </h1>

- Its an interactive system that can provide concise information from the given directory(which ideally should have pdf files in them).
- NOT an original model. Its based on <code>meta-llama/Llama-2-7b-chat-hf</code> model and embedded with <code>sentence-transformers/all-mpnet-base-v2</code> model.
- create virtual env with <code>requirements.txt</code> and run <code>python info_agg.py --doc_path 'document dir'</code>.
- Return <code>quit</code> in place of question to stop.
- Time for 1 Inference depends on the complexity of the question and depth of the answer; It is tested on Tesla V100 GPU
- For reference an example directory is given 3 james bond books :')
<h2>For customized usage and better experience</h2> 
<p>please provide information that are not readily available in internet or even private data. It doesn't hold any information so its completely safe as well. </p>

<h2> Future Plan for Improvement </h2>
<p>Convert this to a dynamic Library to readily use in edge devices; don't know if possible though :')</p>