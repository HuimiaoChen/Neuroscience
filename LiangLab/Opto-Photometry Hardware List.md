# Opto-Photometry Hardware List

[TOC]

Here lists part of the components we need to build the optogenetics/fiberphotometry system.

Reference: [Nanosec Photometry (original name: NidaqGUI)](https://github.com/xzhang03/NidaqGUI).

## Main body (Nanosec)

This is the main body of the whole system, where we can see signals of licking, rotary encoder, I2C, food, camera, buzzer, etc. Teensy 4.0 board is used.

![Schematics](https://github.com/xzhang03/NidaqGUI/raw/master/PCBs/Nanosec/Schematic_Omnibox%20v3_2023-01-05.png)

Material (including the BOM and Gerber files): https://github.com/xzhang03/NidaqGUI/tree/master/PCBs/Nanosec (use the version with most recent date)

## Buzzer

![Schematic](https://github.com/xzhang03/NidaqGUI/raw/master/PCBs/Buzzer/Schematic_Buzzer_2022-04-13.png)

Material (including the BOM and Gerber files): https://github.com/xzhang03/NidaqGUI/tree/master/PCBs/Buzzer

## Shifter

The shifter is used to further multiplex photometry and optogenetic channels. It takes a pulse train and splits it into 2-4 trains. It is used when we have two or more wavelengths of the photometry light coming in one fiber.

The schematic:

![](https://github.com/xzhang03/NidaqGUI/raw/master/PCBs/Shifter/Schematic_Nanosec%20Shifter_2022-09-27.png)

Material (including the BOM and Gerber files): https://github.com/xzhang03/NidaqGUI/tree/master/PCBs/Shifter

## PCB for LED cue

This cue can be used in both unconditional and conditional behavioral tasks, but cues are optional. (1) In the unconditional scenario, the cue signal is given before the food TTL. (2)In the conditional scenario, the cue signal is given before the action and food windows; then if there is an action (e.g., licking), the food TTL is triggered.

LED cue takes a digital signal or PWM signal and turns LED on/off. Wavelength is determined by the components.

The schematic:

![](https://github.com/xzhang03/NidaqGUI/raw/master/PCBs/LED%20cue/Schematic_LED%20cue_2022-04-19.png)

Material (including the BOM and Gerber files): https://github.com/xzhang03/NidaqGUI/tree/master/PCBs/LED%20cue

## Dimmable RGB module (I2C)

This is an I2C version of LED cue delivery. The PCA9685 board is used.

DIO expander is recommended to be used if using dimmable RGB module.

The schematic:

![Schematic](https://github.com/xzhang03/NidaqGUI/raw/master/PCBs/LED%20cue%20i2c/Schematic_LED%20cue%20I2C_2023-01-03.png)

Material (Gerber files **but no BOM**): https://github.com/xzhang03/NidaqGUI/tree/master/PCBs/LED%20cue%20i2c

## I2C repeater

I2C repeater is a device used to support more I2C devices and allow I2C devices to be placed further apart.

It is recommended to use in the "multiple trial types" scenario where we might have different cues to manage and schedule. Note that cue pulses are optional.

The schematic:

![](https://github.com/xzhang03/NidaqGUI/raw/master/PCBs/I2C%20repeater/Schematic_I2C%20repeater_2023-01-03.png)

Material (Gerber files **but no BOM**): https://github.com/xzhang03/NidaqGUI/tree/master/PCBs/I2C%20repeater

## DIO expander

This board expands the digital capability of nanosec. If the dimmable RGB module is used, the external DIO expander should also be used to indicate trial types/onsets. 

The schematic:

![Schematic](https://github.com/xzhang03/NidaqGUI/raw/master/PCBs/DIO%20expander/Schematic_Nanosec%20DIO_2023-01-03.png)

Material (Gerber files **but no BOM**): https://github.com/xzhang03/NidaqGUI/tree/master/PCBs/DIO%20expander

## Hookup between Nanosec, dimmable RGB module, DIO expander and I2C repeater.

![Hookup scheme](https://github.com/xzhang03/NidaqGUI/raw/master/Schemes/Multi%20trialtype%20hookup%20guide.png)

## Others

[Containment for PCB box](https://github.com/xzhang03/Half_breadboard_box/blob/main/half breadboard box PCB.ai)

[Breadboard for Omniphotometrybox](https://github.com/xzhang03/NidaqGUI/blob/master/Schemes/omnibox_half_breadboard.png).