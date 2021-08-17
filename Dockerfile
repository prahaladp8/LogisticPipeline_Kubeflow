FROM prahaladp8/lrbase:3.0
RUN rm -rf LR-Prepare 
RUN rm -rf LR 
RUN mkdir LR-Prepare
WORKDIR /LR-Prepare
#RUN cd LR-Prepare
#COPY requirements.txt ./

#RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get install -y git
RUN git clone https://github.com/prahaladp8/LogisticPipeline_Kubeflow.git
RUN cd LogisticPipeline_Kubeflow
WORKDIR /LR-Prepare/LogisticPipeline_Kubeflow
RUN mkdir input
RUN cd input
RUN wget 'https://www.dropbox.com/s/tq3xz0piqitnc59/loan_data_2007_2014.csv?dl=0' -O ./LR-Prepare/LogisticPipeline_Kubeflow/input/train.csv
RUN wget 'https://www.dropbox.com/s/z77a2qwch6xgsy3/loan_data_2015.csv?dl=0' -O ./LR-Prepare/LogisticPipeline_Kubeflow/input/test.csv
WORKDIR /LR-Prepare/LogisticPipeline_Kubeflow

#RUN mkdir base
#RUN mkdir inputs
#RUN mkdir outputs
#COPY base base
#COPY Driver.py .
#CMD ["python", "Driver.py","--configpath","PipelineInputs.py","--outputpath","op"] 


# Pass data from the container/pod to the outputs directory and
# Pass outputs directory path to next container / pod output
# Copy prev container/pod output to this container / pod's input.