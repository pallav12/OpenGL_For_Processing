import pallav.Matrix.*;
class Model{
  int input_nodes;
  int hidden_nodes; 
  int output_nodes;
  
  Matrix weights_ih;
  Matrix weights_ho;
  
  Matrix bias_h;
  Matrix bias_o;
  
  public Model(int in_nodes,int hid_nodes, int out_nodes){
    this.input_nodes = in_nodes;
      this.hidden_nodes = hid_nodes;
      this.output_nodes = out_nodes;
      this.weights_ih=Matrix.array(new float[hidden_nodes][input_nodes]);
      this.weights_ho = Matrix.array(new float[output_nodes][hidden_nodes]);;
      randomize(weights_ih);
      randomize(weights_ho);
      
      

      this.bias_h = Matrix.array(new float[this.hidden_nodes][1]);
      this.bias_o = Matrix.array(new float[this.output_nodes][1]);  
      randomize(bias_h);
      randomize(bias_o);
      
      
  }
  
  public float[][] convertToMatrix(int[] a){
    float[][] frr=new float[a.length][1];
    for(int i=0;i<a.length;i++){
     frr[i][0]=a[i]; 
    }
    return frr;
  }
  public void train(int[] input_array, int[] target_array){
    Matrix inputs=Matrix.array(convertToMatrix(input_array));
    Matrix hidden=Matrix.Multiply(weights_ih,inputs);
   hidden= Matrix.add(hidden,bias_h);
    
    map(hidden);
    Matrix outputs=Matrix.Multiply(weights_ho,hidden);
    outputs=Matrix.add(outputs,bias_o);
    map(outputs);
    
    Matrix target=Matrix.array(convertToMatrix(target_array));
    Matrix.print(target);
    Matrix.print(outputs);
    Matrix output_error=Matrix.subtract(target,outputs);
    Matrix.print(output_error);
    Matrix gradients=dmap(outputs);
    dotMultiply(gradients,output_error);
    dotMultiply(gradients,0.1f);
    
    Matrix hidden_T=transpose(hidden);
    Matrix weight_ho_deltas=Matrix.Multiply(gradients,hidden_T);
    
    weights_ho=Matrix.add(weights_ho,weight_ho_deltas);
    bias_o=Matrix.add(bias_o,gradients);
    
    Matrix who_t=transpose(weights_ho);
    
    Matrix hidden_errors=Matrix.Multiply(who_t, output_error);
    
    Matrix hidden_gradient=dmap(hidden);
    
    dotMultiply(hidden_gradient, hidden_errors);
    dotMultiply(hidden_gradient, 0.1f);
    
    Matrix input_t=transpose(inputs);
    Matrix weight_ih_deltas=Matrix.Multiply(hidden_gradient,input_t);
    weights_ih=Matrix.add(weights_ih,weight_ih_deltas);
    bias_h=Matrix.add(bias_h, hidden_gradient);
  }
  
   public Matrix transpose(Matrix mat1){
     Matrix transpose=Matrix.array(new float[Matrix.dimensions(mat1)[1]][Matrix.dimensions(mat1)[0]]);
    for(int i =0;i< Matrix.dimensions(transpose)[0];i++){
      for(int j=0;j<Matrix.dimensions(transpose)[1];j++){
        transpose.array[i][j]=mat1.array[j][i];
  }
    }  
    return transpose;
  }
  
  public Matrix dmap(Matrix mat){
    for(int i =0;i< Matrix.dimensions(mat)[0];i++){
      for(int j=0;j<Matrix.dimensions(mat)[1];j++){
        mat.array[i][j]=mat.array[i][j]*(1-mat.array[i][j]);
  }
    }  
    return mat;
  }
  
  public Matrix dotMultiply(Matrix mat1,Matrix mat2){
    for(int i =0;i< Matrix.dimensions(mat1)[0];i++){
      for(int j=0;j<Matrix.dimensions(mat1)[1];j++){
        mat1.array[i][j]=mat1.array[i][j]*mat2.array[i][j];
  }
    }  
    return mat1;
  }
    public Matrix dotMultiply(Matrix mat1,float mat2){
    for(int i =0;i< Matrix.dimensions(mat1)[0];i++){
      for(int j=0;j<Matrix.dimensions(mat1)[1];j++){
        mat1.array[i][j]=mat1.array[i][j]*mat2;
  }
    }  
    return mat1;
  }
  
  
  public float[][] predict(int[] a){
    
    Matrix inputs=Matrix.array(convertToMatrix(a));


    Matrix hidden=Matrix.Multiply(weights_ih,inputs);
    hidden=Matrix.add(hidden,bias_h);
    map(hidden);
    
    Matrix output=Matrix.Multiply(weights_ho,hidden);
    output=Matrix.add(output,bias_o);
    
    map(output);
    return output.array;
    
  }
  
  private void randomize(Matrix mat){
    for(int i =0;i< Matrix.dimensions(mat)[0];i++){
      for(int j=0;j<Matrix.dimensions(mat)[1];j++){
        mat.array[i][j]=random(-1,1);
  }
    }
    }
    
    private void map(Matrix mat){
       for(int i =0;i< Matrix.dimensions(mat)[0];i++){
      for(int j=0;j<Matrix.dimensions(mat)[1];j++){
        mat.array[i][j]=sigmoid(mat.array[i][j]);
  }
    }
    }
    private float sigmoid(float a){
      return 1/(1+exp(-a));
      
    }
    
}
