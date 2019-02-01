import { Component, OnInit, ViewChild } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {

  linearModel: tf.Sequential;
  prediction: any;

  ngOnInit() {
    this.train();
  }


  async train(): Promise<any> {
    // Lineáris regresszió modell definiálása.
    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({units: 1, inputShape: [1]}));

    // Modell előkészítése tanításra: Loss és optimalizáció definiálása.
    this.linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});


    // Modell tanítása véltlenszerű értékekkel.
    const xs = tf.tensor1d([3.2, 4.4, 5.5]);
    const ys = tf.tensor1d([1.6, 2.7, 3.5]);


    // Tanítás
    await this.linearModel.fit(xs, ys);

    console.log('model trained!');
  }

  predict(val: number) {
    const output = this.linearModel.predict(tf.tensor2d([val], [1, 1])) as any;
    this.prediction = Array.from(output.dataSync())[0];
  }}
