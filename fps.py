from datetime import datetime

class FPS:
    """A super simple class for measuring frames per second.
    Usage: 
	Call incrementFrames() every time a frame is shown.
    	Call getFPS() to get the current frames per second.
        Optionally, call reset() to restart the count."""

    def __init__(self):
        self.reset()

    def getFPS(self):
        "Return the current frames per second as a float"
        return self.framecount / (datetime.now() - self.start).total_seconds()
    
    def incrementFrames(self):
        "Main loop should call this every time a frame is shown" 
        self.framecount += 1

    def reset(self):
        "Restart the stopwatch"
        self.start = datetime.now()
        self.framecount = 0
        
