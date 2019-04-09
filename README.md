# AMC
Adversarial Malware Crafter
The two files include the program to vary the number of Cache Misses and Branch Instructions, as required by the ML classifier.

In the Branhc_Instruction, multiple nested loops have been created to increase the number of branch instructions and branch misses. 
Some paramenets such as loop size, number of iterations were calculated using brute force method, to meet the required values.

In the clflush file, the code is written to vary the number of cache misses. With the help of using Embedded Clflush instructions,
the elements of the declared array are removed respectively from the cache memory and then reloaded to vary the number of cache misses.
(Explained more in the code)
