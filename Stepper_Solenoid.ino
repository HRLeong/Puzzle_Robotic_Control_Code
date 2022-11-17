#include <string.h> 
const int DIR = 2;
const int STEP = 3;
const int SOL = 4;
const int  steps_per_rev = 400;
const char *delimiter = "_";

void setup()
{
  Serial.begin(115200);

  pinMode(STEP, OUTPUT);
  pinMode(DIR, OUTPUT);
  pinMode(SOL, OUTPUT);

  digitalWrite(DIR, LOW);
}

void loop()
{ 
  if (Serial.available() > 0) {
    String input_string = Serial.readStringUntil('\n');
    if (input_string == "SOL_OPEN") {
      digitalWrite(SOL, HIGH);
      Serial.println("Solenoid Valve Opened");
    } else if (input_string == "SOL_CLOSE") {
      digitalWrite(SOL, LOW);
      Serial.println("Solenoid Valve Closed");
    } else if (input_string == "ROTATE") {
      while (input_string != "STOP") {
        digitalWrite(STEP, HIGH);
        delay(50);
        digitalWrite(STEP, LOW);
        delay(50);
        if (Serial.available() > 0) {
          input_string = Serial.readStringUntil('\n');
        }
      }
    } else {
      Serial.println("INVALID INPUT");
    }
    Serial.println("DONE");
  }
}
