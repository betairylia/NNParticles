import os
import sys

for subdir, dirs, files in os.walk('.'):
    for filep in files:
        if filep.split('.')[-1] == "off":
            filep = os.path.join(subdir, filep)

            # Convert the **** ModelNet un-formal format ...
            with open(filep, 'r') as file_in:
                with open(filep + '.tmp', 'w') as file_out:
                    fLine = file_in.readline()
                    file_out.write(fLine[:3] + '\n')
                    file_out.write(fLine[3:] + '\n')
                    for line in file_in.readlines():
                        file_out.write(line + '\n')

            command_str = "off2obj %s.tmp -o %s.obj" % (filep, filep)
            print(command_str)
            os.system(command_str)

            command_str = "rm %s.tmp" % (filep)
            print(command_str)
            os.system(command_str)

