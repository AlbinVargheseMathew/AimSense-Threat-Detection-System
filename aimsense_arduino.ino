#include <Servo.h>

Servo servoX;
Servo servoY;

int laserPin = 7;

String data;

void setup()
{
  Serial.begin(115200);

  servoX.attach(5);
  servoY.attach(6);

  pinMode(laserPin, OUTPUT);
  digitalWrite(laserPin, LOW);
}

void loop()
{

  if(Serial.available()){

    data = Serial.readStringUntil('\n');

    if(data == "FIRE")
    {
      digitalWrite(laserPin, HIGH);
      delay(2000);
      digitalWrite(laserPin, LOW);
    }
    else
    {

      int comma = data.indexOf(',');  

      int x = data.substring(0,comma).toInt();
      int y = data.substring(comma+1).toInt();

      servoX.write(x);
      servoY.write(y);

    }

  }

}