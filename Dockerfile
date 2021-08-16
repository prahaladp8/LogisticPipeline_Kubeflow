FROM prahaladp8/lrbase:3.0
RUN rm -rf LR-Prepare 
RUN mkdir LR-Prepare
WORKDIR /LR-Prepare
#RUN cd LR-Prepare
#COPY requirements.txt ./

#RUN pip install --no-cache-dir -r requirements.txt

#RUN apt-get install -y git
#RUN mkdir /root/.ssh/
#ADD id_rsa /root/.ssh/id_rsa
# Create known_hosts
#RUN touch /root/.ssh/known_hosts
# Add bitbuckets key
#RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
# Clone the conf files into the docker container
#RUN git clone git@bitbucket.org:User/repo.git

RUN mkdir base
RUN mkdir inputs
RUN mkdir outputs
COPY base base
COPY Driver.py .
#CMD ["python", "Driver.py","--configpath","PipelineInputs.py","--outputpath","op"] 


# Pass data from the container/pod to the outputs directory and
# Pass outputs directory path to next container / pod output
# Copy prev container/pod output to this container / pod's input.