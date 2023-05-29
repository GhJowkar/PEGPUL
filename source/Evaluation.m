% PEGPUL: Perceptron ensemble of graph-based positive-unlabeled learning
% Created by: Gholamhossein Jowkar
% Created date: Jan 2015
% Modified by: Gholamhossein Jowkar
% Modified date: 

function EVAL = Evaluation(ACTUAL,PREDICTED)
% This fucntion evaluates the performance of a classification model by 
% calculating the common performance measures: Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures


idx = (ACTUAL()==1);

p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;

tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;

tp_rate = tp/p;
tn_rate = tn/n;

accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);


n = tn+tp+fn+fp;
s = (tp+fn)/n;
p =(tp+fp)/n;
MCC_2 = ((tp/n)-(s*p))/sqrt((p*s)*(1-s)*(1-p));

MCC = ((tp*tn)-(fp*fn))/sqrt((tp+fn)*(tp*fn)*(tn*fp)*(tn+fn));


% EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
% EVAL = [precision recall f_measure];
EVAL = [precision recall f_measure  MCC_2];

