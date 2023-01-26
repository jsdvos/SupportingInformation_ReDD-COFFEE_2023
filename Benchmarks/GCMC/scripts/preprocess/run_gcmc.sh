#!/bin/bash

if [ -d Output ]
then

    simulate -i restart.input >> raspa.log 2>> raspa.err

else
  
    simulate -i simulation.input > raspa.log 2> raspa.err

fi
