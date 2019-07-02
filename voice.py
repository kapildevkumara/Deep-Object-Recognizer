import pyttsx3				  		# Voice Support : Text to Voice

def voice_support(message):
	engine = pyttsx3.init() 			# Object creation
	engine.setProperty('rate', 140)     	        # Voice rate
	engine.setProperty('volume', 0.8)    	        # Setting up volume level between 0 and 1
	engine.setProperty('voice', voices[9].id)       # changing index, changes voices. 1 for Females and 0 for Males

	#engine.say("You Have" + " an Obstacle at " + " 1 Metre" )
	engine.say("You Have" + string(message[0]) + " at " +  string(message[1]) + "Metre")		# Pass Message as a List of Length 2 with 1st Message - Object Name and 2nd Message - Coordinates
	engine.runAndWait()
	engine.stop()
	return 1

