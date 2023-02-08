# *******************  IMPORTING MODULES ********************
from datetime import datetime
from pythonosc import dispatcher
from pythonosc import osc_server
from timeit import default_timer as timer

# *********************  G L O B A L S *********************
alpha = beta = delta = theta = gamma = [-1,-1,-1,-1]
all_waves = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

# ip = "192.168.1.255"  # for brodcat
ip = "0.0.0.0"  # for localhost

port = 9998
dispatcher = dispatcher.Dispatcher()

filePath = 'Blinks/Regular.csv'
filePath2 = 'Blinks/'

f = open (filePath,'a+')
header = 'timestamp,TP9_D,TP9_T,TP9_A,TP9_B,TP9_G,AF7_D,AF7_T,AF7_A,AF7_B,AF7_G,AF8_D,AF8_T,AF8_A,AF8_B,AF8_G,TP10_D,TP10_T,TP10_A,TP10_B,TP10_G\n'

current_file = ''
current_event = 0
row = 0

secs = 2
start = timer()
recording = False

# Recording EEG-data: 
# Show an event to record for a predefined time, this is preferable as it creates files that directly can be uploaded to Edge Impulse

record_many = True
                                       
# Put the events to record in this dictionary within "" and after : the seconds
rec_dict = {
    "Blink"   : 2,
    "Regular" : 2
}  


# ==========================================================
# *******************  F U N C T I O N S *******************
# ==========================================================


# ****************** EEG-handlers ******************

def EEG_handler(address: str,*args):
    global record_many

    if (len(args)==21):                                              # If OSC Stream Brainwaves = All Values
        for i in range(1,21):
            all_waves[i-1] = args[i]

    if record_many == False:
        for i in range(0,19):
            f.write(str(all_waves[i]) + ",")
        f.write(str(all_waves[19]))
        f.write("\n")
    else:
        show_event()

# ********* Showing one event at a time *********
def show_event():
    global current_event, current_file
    global start, end, secs, row

    end = timer()
    if (end - start) >= secs:                                       # if we've waited enough for the current event
        start = timer()                                             # getting current time
        ev = list(rec_dict.items())[current_event][0]               # fetching current event
        secs = list(rec_dict.items())[current_event][1]             # fetching seconds for current event
        row = 0
        
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%Y-%m-%d %H_%M_%S.%f")

        ev = list(rec_dict.items())[current_event][0]
        current_file = filePath2 + ev + '.' + timestampStr + '.csv'
        evf = open (current_file,'a+')
        evf.write(header)


        print(f"Think:\t {ev}   \t\t{secs}  seconds")

        dict_length = len(rec_dict)                                 # how many events in the dictionary
        if current_event < dict_length-1:                           # if end not reached...
            current_event += 1                                      # ...increasing counter
        else:
            current_event = 0                                       # if end reached, starting over
    else:
        if current_file != '':
            evf = open (current_file,'a+')

            evf.write(str(row) + ',')
            row += 1
            for i in range(0,19):
                evf.write(str(all_waves[i]) + ",")
            evf.write(str(all_waves[19]))
            evf.write("\n")


# def marker_handler(address: str,i):
#     global recording, record_many, start, end

#     dateTimeObj = datetime.now()
#     timestampStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
#     markerNum = address[-1]
#     f.write(timestampStr+",,,,/Marker/"+markerNum+"\n")
#     start = timer()
#     if (markerNum=="1"):        
#         recording = True
#         print("Recording Started.")
#     if (markerNum=="2"):
#         f.close()
#         server.shutdown()
#         print("Recording Stopped.") 

#     if (markerNum=="3"):
#         start = timer()

#         for i in range(len(rec_dict)):
#             ev = list(rec_dict.items())[i][0]
#             evf = open (filePath2 + ev + '.csv','a+')
#             evf.write(header)

#         if record_many == False:
#             record_many = True
#             show_event()
#         else:
#             record_many = False


if __name__ == "__main__":
    

    # dispatcher.map("/Marker/*", marker_handler)
    dispatcher.map("/allwaves", EEG_handler,0)
    
    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    # print("Listening on UDP port "+str(port)+"\nSend Marker 1 to Start recording, Marker 2 to Stop Recording, Marker 3 to show events. ")
    # print()
    server.serve_forever()