def upscale(name): 
    import cv2
    import numpy as np
    from gaussian2d import gaussian2d
    from scipy import interpolate
    
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule    
    
    drv.init()
    device = drv.Device(0) # enter your gpu id here
    ctx = device.make_context()
    mod = SourceModule("""
    #include"math.h"
    __device__ void hashkey(float block[9][9],int Qangle,float*weight,int*r)
    {
        int i,j,k;
        float gy[81];
        for(j=0;j<9;j++)
        {
            gy[j]=block[1][j]-block[0][j];
            gy[72+j]=block[8][j]-block[7][j];
        }
          
        for(i=1;i<8;i++){
                for(j=0;j<9;j++){
                        gy[i*9+j]=(block[i+1][j]-block[i-1][j])/2.0;
                }
        }   
              
        float gx[81];
        for(i=0;i<9;i++)
        {
            gx[i]=block[i][1]-block[i][0];
            gx[72+i]=block[i][8]-block[i][7];
        }
          
        for(i=0;i<9;i++){
                for(j=1;j<8;j++){
                        gx[i*9+j]=(block[i][j+1]-block[i][j-1])/2.0;
                }
        }      
              
        float g[2][81];
        float gt[81][2];
        for(i=0;i<81;i++)
        {
            g[0][i]=gx[i];
            g[1][i]=gy[i];
            gt[i][0]=gx[i];
            gt[i][1]=gy[i];
        }
        float gtwg[2][2];
        for(i=0;i<2;i++){
                for(j=0;j<81;j++){
                        g[i][j]=g[i][j]*weight[j*81+j];
                }
        }
             
        for(i=0;i<2;i++){
                for(j=0;j<2;j++){
                        for(k=0;k<81;k++){
                                gtwg[i][j]+=g[i][k]*gt[k][j];
                        }
                }
        }           
                  
        float w[2];
        float v[2][2];        
        float disc=(gtwg[0][0]+gtwg[1][1])*(gtwg[0][0]+gtwg[1][1])-4*(gtwg[0][0]*gtwg[1][1]-gtwg[0][1]*gtwg[1][0]);
        if(disc<0)
        {
            w[0]=(gtwg[0][0]+gtwg[1][1])/2;
            w[1]=(gtwg[0][0]+gtwg[1][1])/2;
        }
        else
        {
            w[0]=(gtwg[0][0]+gtwg[1][1])/2+sqrt(disc)/2;
            w[1]=(gtwg[0][0]+gtwg[1][1])/2-sqrt(disc)/2;
        }
        
        v[1][0]=(w[0]-gtwg[1][1])/gtwg[1][0];
        v[0][0]=1;
        v[1][1]=(w[1]-gtwg[1][1])/gtwg[1][0];
        v[0][1]=1;
        float theta=atan2(v[1][0],v[0][0]);
        if(theta<0){
                theta=theta+M_PI;
        }     
      
        float lamda = w[0];
    
        float sqrtlamda1 = sqrt(w[0]);
        float sqrtlamda2 = sqrt(w[1]);
        int u;
        if(sqrtlamda1 + sqrtlamda2 == 0){
                u = 0;
        } 
        else{
                u = (sqrtlamda1 - sqrtlamda2)/(sqrtlamda1 + sqrtlamda2);
        }
        int angle = floor(theta/M_PI*Qangle);
        int strength,coherence;
        if (lamda < 0.0001){
                strength = 0;
        }
        else if(lamda > 0.001){
                strength = 2;
        }  
        else{
                strength = 1;
        }
            
        if (u < 0.25){
                coherence = 0;
        }  
        else if(u > 0.5){
                coherence = 2;
        }   
        else{
                coherence = 1;
        }
            
    
        if (angle > 23){
                angle = 23;
        }   
        else if(angle < 0){
                angle = 0;
        }
            
        r[0]=int(angle);
        r[1]=int(strength);
        r[2]=int(coherence);
    }
        
    __global__ void patch_up(float *predictHR, float *a, float *h,float *w,float*shape)
    {
        int height=(int)shape[0];
        int width=(int)shape[1];
        int i = 5+blockIdx.x * blockDim.x + threadIdx.x; 
        int j = 5+blockIdx.y * blockDim.y + threadIdx.y;
        if(i>=height-5 || j>=width-5)
        {
                return;
        }
        
        float patch[121];
        int row,col;
        
        for(row=0;row<11;row++){
                for(col=0;col<11;col++){
                        patch[row*11+col]=a[(i-5+row)*width+(j-5+col)];
                }
        }        
    
        float block[9][9];
        for(row=0;row<9;row++){
                for(col=0;col<9;col++){
                        block[row][col]=a[(i-4+row)*width+j-4+col];
                }
        }
        
        int r[3];    
        hashkey(block, 24, w,r);
        int pixeltype = ((i-5) % 2) * 2 + ((j-5) % 2);
        float hh[121];
        int hUsed=r[0]*36*121+r[1]*12*121+r[2]*4*121+pixeltype*121;
        for(row=0;row<121;row++){
                hh[row]=h[hUsed+row];
        }
        
        int position=(i-5)*(width-10)+j-5;
        for(row=0;row<121;row+=1){
                predictHR[position]+=patch[row]*hh[row];
        }
            
    }
    """)
            

    # Define parameters
    gradientsize = 9
    margin = 5
    
    weighting = gaussian2d([gradientsize, gradientsize], 2)
    weighting = np.diag(weighting.ravel())#81*81
    #with open("filter_BSDS500", "rb") as fp:
        #h = pickle.load(fp)
    #h=np.array(h)
    h=np.load('filter.npy')
    
    img = cv2.imread('./static/' + name)
    size = img.shape
    origin = cv2.resize(img,(int(size[1]/2),int(size[0]/2)),cv2.INTER_LINEAR)
    #cv2.imwrite('./static/LR_' + name, origin)

    ycrcvorigin = cv2.cvtColor(origin, cv2.COLOR_BGR2YCrCb)
    grayorigin = ycrcvorigin[:,:,0]
    grayorigin = cv2.normalize(grayorigin.astype('float'), None, grayorigin.min()/255, grayorigin.max()/255, cv2.NORM_MINMAX)
    heightLR, widthLR = grayorigin.shape
    heightgridLR = np.linspace(0,heightLR-1,heightLR)
    widthgridLR = np.linspace(0,widthLR-1,widthLR)
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, grayorigin, kind='linear')
    heightgridHR = np.linspace(0,heightLR-0.5,heightLR*2)
    widthgridHR = np.linspace(0,widthLR-0.5,widthLR*2)
    upscaledLR = bilinearinterp(widthgridHR, heightgridHR).astype(np.float32)
    heightHR, widthHR = upscaledLR.shape

    upscaledLR=upscaledLR.reshape((heightHR*widthHR))
    predictHR = np.zeros(((heightHR-2*margin)*(widthHR-2*margin))).astype(np.float32)
    h=h.reshape((24*3*3*4*121)).astype(np.float32)
    weighting = weighting.astype(np.float32).reshape((81*81))
    patch_up = mod.get_function("patch_up")
    grid1=int((heightHR-2*margin)/16)+1
    grid2=int((widthHR-2*margin)/16)+1
    shape=np.array([heightHR, widthHR]).astype(np.float32)
    
    patch_up(drv.Out(predictHR), drv.In(upscaledLR), drv.In(h),drv.In(weighting),drv.In(shape),grid=(grid1,grid2),block=(16,16,1))
    ctx.pop()
    
    predictHR=predictHR.reshape((heightHR-2*margin),(widthHR-2*margin))
    upscaledLR=upscaledLR.reshape((heightHR,widthHR))
    predictHR = np.clip(predictHR.astype('float') * 255., 0., 255.)

    result = np.zeros((heightHR-2*margin, widthHR-2*margin, 3))
    resultBLI = np.zeros((heightHR-2*margin, widthHR-2*margin, 3))
    y = ycrcvorigin[:,:,0]
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, y, kind='linear')
    resultBLI[:,:,0] = bilinearinterp(widthgridHR, heightgridHR)[margin:heightHR-margin,margin:widthHR-margin]

    cr = ycrcvorigin[:,:,1]
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cr, kind='linear')
    result[:,:,1] = bilinearinterp(widthgridHR, heightgridHR)[margin:heightHR-margin,margin:widthHR-margin]
    cv = ycrcvorigin[:,:,2]
    bilinearinterp = interpolate.interp2d(widthgridLR, heightgridLR, cv, kind='linear')
    result[:,:,2] = bilinearinterp(widthgridHR, heightgridHR)[margin:heightHR-margin,margin:widthHR-margin]
    result[:,:,0] = predictHR
    resultBLI[:,:,1:]=result[:,:,1:]
    result = cv2.cvtColor(np.uint8(result), cv2.COLOR_YCrCb2RGB)
    '''
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    result = cv2.filter2D(result, -1, kernel=kernel)
    '''
    resultBLI = cv2.cvtColor(np.uint8(resultBLI), cv2.COLOR_YCrCb2RGB)
    cv2.imwrite('./static/HR_' + name, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite('./static/BLI_' + name, cv2.cvtColor(resultBLI, cv2.COLOR_RGB2BGR))
    if widthHR>heightHR:
        w=800
        h=int((heightHR-2*margin)/((widthHR-2*margin)/800))
    else:
        h=600
        w=int((widthHR-2*margin)/((heightHR-2*margin)/600))
    return h,w

