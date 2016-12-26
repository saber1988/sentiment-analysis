# sentiment-analysis
Sentiment analysis using IMDB data set. The code uses tensorflow to implement a sequence2vec model to generate a paragraph embedding, which is based on the paper: Distributed Representations of Sentences and Documents. Based on the paragraph embedding, the code uses random forest, gbdt and svc to do the sentiment classification. Among these three models, svc works best and gets an accuracy of 0.82940. This is a raw model and a lot of parameter adjustment can be done.

Below is the image of word vectors, using t-sne to reduce the dimensionality to 2    
![this](https://github.com/saber1988/sentiment-analysis/tree/master/img/tsne-word.png)     

Below is the image of paragraph vectors, using t-sne to reduce the dimensionality to 2     
![this](https://github.com/saber1988/sentiment-analysis/tree/master/img/tsne-para.png)     
