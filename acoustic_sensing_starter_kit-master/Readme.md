# Acoustic Sensing Starter Kit

This repository contains example scripts for the "Acoustic Sensing Starter Kit".

Link to Starter Kit: https://www.robotics.tu-berlin.de/menue/software_and_tutorials/acoustic_sensing_starter_kit

![Acoustic Sensing](img/active_acoustic_sensing.png "Acoustic Sensor embedded into a PneuFlex actuator")
Fig. 1: An acoustic sensor consisting of a microphone (left) and a speaker (right) embedded into a PneuFlex actuator

### Related publications:
* Gabriel Zöller, and Vincent Wall and Oliver Brock. [“Active Acoustic Contact Sensing for Soft Pneumatic Actuators.”](http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/Zoeller-20-ICRA_activeacoustic.pdf) IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.
* Gabriel Zöller and Vincent Wall and Oliver Brock. ["Acoustic Sensing for Soft Pneumatic Actuators."](http://www.robotics.tu-berlin.de/fileadmin/fg170/Publikationen_pdf/Zoeller-18-IROS_acousticsensing.pdf) IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 6986–6991, 2018.

### Notes
This guide is for Linux machines, due to the used audio libraries (zita-jacktools). It was tested on Ubuntu 18.04 and Debian 10.
If you replace the audio library, the rest of the code should work on any system with a current Python version. 

### Prerequisites
1. Install `zita-jacktools`
    * run `zita-tools/install.sh`
    * install with `pip install zita-tools/zita-jacktools`
2. Install python requirements from `requirements.txt`.
    * `pip install -r requrirements.txt`
    * If you run into build errors and you're using conda, try one of these:
        * `conda install <package>`
        * `conda install -c conda-forge <package>`

### Running the scripts
1. Open QjackCtl GUI
    * `qjackctl`
    * Check that the settings are set to:       
        * Interface: hw:<your audio interface/headset or even internal microphone/speakers>, e.g. hw:USB MAYA44
	    * Sample Rate: 48000
	    * Frames/Period: 2048
	    * Periods/Buffer: 2
    * Press 'play'.
2. Execute `1_record.py` to record training data:
    * `cd <Acoustic_Sensing_Directory>/` # folders are interpreted relative to this directory while you're running these scripts
	* `python 1_record.py`
	* See current class at the top, e.g. 'base'. Make contact accordingly or whatever you are trying to classify.
	* Click the "Record" button or press 'R' to play the active sound and simultaneously record with the speaker.
	* Repeat until all samples are recorded.
3. Execute `2_train.py` to train a sensor model using the previously recorded audio samples.
    * `python 2_train.py`
4. Execute `3_sense.py` to run a live sensor that continuously playes the active sound and uses the trained sensor model to sense.
    * `python 3_sense.py`
    * The active sound is played in a loop. Each time the trained sensor model predicts the current class.
    * You can disable the `CONTINUOUSLY` flag in the code, to manually trigger the sensing with `<Enter>` on the console. 
5. Be amazed at the surprising success of acoustic sensing! :-) 

### Contact
Vincent Wall
https://www.robotics.tu-berlin.de/menue/team/vincent_wall/

```
@author: Vincent Wall, Gabriel Zöller
@copyright 2020 Robotics and Biology Lab, TU Berlin
@licence: BSD Licence
```