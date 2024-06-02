import serial
import time

#ser = serial.Serial('/dev/ttyACM0', timeout=0.01)  # open serial port
         # check which port was really used

for inx_time in range(10):
    
    ser = serial.Serial('/dev/ttyACM0', timeout=0.01)
    print(ser.name)

    try:
        ser.write(b'\r')     # write a string
    except ser.SerialTimeoutException:
        print('TimeOut')

    try:
        print(ser.read_all())
    except ser.SerialTimeoutException:
        print('TimeOut')
       
    time.sleep(1)
    
    ser.close()
    # print('Proving...')
    # time.sleep(1)
    # try:
    #     print(ser.read(50))
    # except ser.SerialTimeoutException:
    #     print('TimeOut')
    #     ser.close()             # close port
    
ser.close()             # close port
