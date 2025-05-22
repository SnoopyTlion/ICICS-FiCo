1. To run our code, you need download the code first;
2. In `global_vars.py`, you need change some places to adopt your own file:
   1) In line 18, we use an on-the-fly bert tokenizer, which can be downloaded from https://huggingface.co/google-bert/bert-base-uncased;
   2) In line 19, we initialize the classifier, you can find it in fico/model;
   3) In line 33, we use the limited keyword list from the keyword-dependent approach as a start point for affinity analysis.
   4) In line 9, we assign the tested app's package name for analysis.
3. Run following command with pip:
```
pip install -r requirements.txt
```
4.  Then, run ```python3 app_explorer.py``` to start the analysis.
In our example, we use TikTok to run the test, we install TikTok (com.zhiliaoapp.musically) in a Pixel phone with uiautomator2 environment.
5. The log file will be stored in `./logdir`.
