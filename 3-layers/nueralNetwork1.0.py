#using vscode to make your own neutral network
#neutral network class definition
#enable for plotting arrays
import matplotlib.pyplot
import numpy
import pylab
import scipy.special
from numpy.core.numeric import allclose
from numpy.lib.type_check import asfarray


#calss definition
class neutralNetwork :

    #initialise the neutral network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learingrate) :

        #set number of the nodes in each input,hidden,output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #link weight matrices,wih and who
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #set learingrate
        self.lr = learingrate
        #activation function is the sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    #train the neutral network
    def train(self,inputs_list,targets_list):
        #convert input list and target list to 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T
        target = numpy.array(targets_list,ndmin=2).T
        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        #calculate signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        #calculate signals into final layer
        final_input = numpy.dot(self.who,hidden_outputs)
        #calculate signals emerging from the final layer
        final_output = self.activation_function(final_input)
        #calculate the ouput_error,which is targets-final_ouput
        output_error = target - final_output
        #hidden error is the output-error,split by weights
        hidden_error = numpy.dot(self.who.T,output_error)
        #update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_error * final_output * (1.0 - final_output)),numpy.transpose(hidden_outputs))
        #update the weights dor the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_error * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
        pass
    #query the neutral network
    def query(self,input_list):
        
        #convert input_list to 2d array
        inputs = numpy.array(input_list,ndmin=2).T
        #calculate the signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        #calculate the signals emerging from hidden_layer
        hidden_outputs = self.activation_function(hidden_inputs)
        #calculate the signals to final ouput layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
#number for input,hidden,output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

#learning rate is 0.1  the experiment show learning rate between 0.1 and 0.3 will have the best performance
learning_rate = 0.2
#creating instance of neutral network
n = neutralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#load the minist traning data CSV file into a list
tranning_data_file = open("mnist_train.csv",'r')
#must be distinguish readline and readlines,the former means you will read the next line;but the behind means you can load the whole file
tranning_data_list = tranning_data_file.readlines()
tranning_data_file.close()

#tranning the neutral network
for e in range(5):
    for record in tranning_data_list:
        #split the record by the ',' commas
        all_values = record.split(',')
        #scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99)+0.01
        #creating the target output value 
        targets = numpy.zeros(output_nodes)+0.01
        #all_value[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        pass
#load the minist test data CSV file into a list
test_data_file = open("mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard = []

for record in test_data_list:

    #split the record by the ',' commas
    all_values = record.split(',')
    #get the target value what you want
    correct_label = int(numpy.asfarray( all_values[0] ))
    #scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:])/255.0 *0.99) +0.01
    #testing the neutral network
    outputs = n.query(inputs)
    #gain the bigest element in your array,and return the index of that
    label = numpy.argmax(outputs)
    #fulfill the scorecard array to calculate the performance
    if (label == correct_label):
        #if the index equal with the target label,insert 1 to scorecard
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass 
pass
#turn string to double
scorecard_array = numpy.asfarray(scorecard)
#calculate the performance,which is the sum / sizes
print("performance = ",(scorecard_array.sum() / scorecard_array .size)*100,'%')

