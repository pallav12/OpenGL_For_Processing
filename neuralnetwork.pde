import pallav.Matrix.*;

void setup(){
 Model model = new Model(2,2,1);
 int[][] train_input={{1,0}};
 int[][]train_output={{1}};
 
 int[] arr ={1,0};
 
   for(int i=0;i<1;i++){
       int rand=(int)random(0);
       model.train(train_input[rand],train_output[rand]);
       float[][] output=model.predict(arr);
       Matrix.print(output);
   }
     
 
 
}

void draw(){
  
  
}
