
class learner():
    def get_acquisition_actor(self):
        raise NotImplementedError

    def updateAcquisitionAgent(self,acquisitionAgent):
        raise NotImplementedError

    def train(self,workspace):
        raise NotImplementedError