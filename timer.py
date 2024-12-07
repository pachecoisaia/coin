import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def start(self):
        """Start the timer."""
        if not self.running:
            self.start_time = time.time() - self.elapsed_time  # Adjust for any previous time
            self.running = True
            print("Timer started...")

    def stop(self):
        """Stop the timer and print the elapsed time."""
        if self.running:
            self.elapsed_time = time.time() - self.start_time  # Calculate elapsed time
            self.running = False
            self.print_elapsed_time()

    def print_elapsed_time(self):
        """Print the elapsed time in Days:Hours:Minutes:Seconds:Milliseconds format."""
        days = int(self.elapsed_time // (24 * 3600))  # Days
        hours = int((self.elapsed_time % (24 * 3600)) // 3600)  # Hours
        minutes = int((self.elapsed_time % 3600) // 60)  # Minutes
        seconds = int(self.elapsed_time % 60)  # Seconds
        milliseconds = int((self.elapsed_time % 1) * 1000)  # Milliseconds

        print(f"Elapsed time: {days:02} days: {hours:02} hours: {minutes:02} minutes: {seconds:02} seconds: {milliseconds:03} milliseconds")