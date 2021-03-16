import torch
from robotStateTransformation import robotStateTransformation

if __name__ == '__main__':
	# example absolute state
	position = [0,0,0]
	quaternion = [0,0,0,1]
	bodyVelocity = [0.1]*6 # velocity of robot relative to its frame
	absoluteState = torch.tensor(position+quaternion+bodyVelocity)
	state = robotStateTransformation(absoluteState)
	print("absoluteState")
	print(state.currentState) # absolute state
	print("pose invariant state")
	print(state.getPredictionInput()) # pose invariant state (up direction relative to robot (R3), body velocity (R6))

	# example transformation
	relativePosition = [1,1,1]
	relativeQuaternion = [0,0,0,1]
	newBodyVelocity = [0.3]*6
	transformation = torch.tensor(relativePosition+relativeQuaternion+newBodyVelocity)
	state.updateState(transformation)
	print(state.currentState)