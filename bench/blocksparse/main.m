// Single-layer microbench for bs.metal (block-sparse fine attention) vs MPSGraph dense SDPA.
// Selection mirrors FastVideo 8db2d71: k = max(1,min(ceil((1-sparsity)*nv),nv)) (video_sparse_attn.py#L165-167),
// per head & query tile top-k of video key tiles by pooled score q_p.k_p/sqrt(D); prefix (non-video)
// query tiles dense; video queries keep all prefix key tiles ("exempt", video_sparse_attn_h3.py#L572-619).
// Usage: ./bs [S] [H] [prefix_tiles] [sparsity]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <QuartzCore/QuartzCore.h>
#include <math.h>
#include <stdlib.h>
typedef struct { uint32_t S, H; float scale_log2; uint32_t QT, NSEL; } bs_args;
static float bf2f(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
static uint16_t f2bf(float f){uint32_t u;memcpy(&u,&f,4);u+=0x7fff+((u>>16)&1);return u>>16;}
static float frand(void){return (float)rand()/RAND_MAX*2.f-1.f;}
static int64_t gscore_n; static const float *gscore;
static int cmpd(const void*a,const void*b){float x=gscore[*(const uint32_t*)a],y=gscore[*(const uint32_t*)b];return x<y?1:x>y?-1:0;}
int main(int argc,char**argv){@autoreleasepool{
  uint32_t S=argc>1?atoi(argv[1]):5952, H=argc>2?atoi(argv[2]):56, P=argc>3?atoi(argv[3]):4, D=128;
  float sparsity=argc>4?atof(argv[4]):0.9f;
  uint32_t QT=(S+63)/64, NV=QT-P;
  uint32_t K=(uint32_t)ceilf((1-sparsity)*NV); if(K<1)K=1; if(K>NV)K=NV;
  const int REPS=10;
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q=[dev newCommandQueue];
  size_t n=(size_t)H*S*D, bytes=n*2;
  id<MTLBuffer> bQ=[dev newBufferWithLength:bytes options:0],bK=[dev newBufferWithLength:bytes options:0],
    bV=[dev newBufferWithLength:bytes options:0],bR=[dev newBufferWithLength:bytes options:0],bO=[dev newBufferWithLength:bytes options:0];
  srand(1); uint16_t*pq=bQ.contents,*pk=bK.contents,*pv=bV.contents;
  // give tiles some structure so top-k is not a coin flip: per-tile offset direction
  for(size_t h=0;h<H;h++)for(size_t r=0;r<S;r++){float tb=sinf((r/64)*0.7f+h);
    for(size_t d=0;d<D;d++){size_t i=(h*S+r)*D+d; float dir=(d%2?1:-1)*tb;
      pq[i]=f2bf(frand()*2+dir); pk[i]=f2bf(frand()*2+dir); pv[i]=f2bf(frand());}}
  float scale=1.0f/sqrtf((float)D);
  // --- selection from pooled scores (CPU; in-engine this comes from the coarse kernel) ---
  uint32_t NSEL=QT+1; uint32_t *sel=calloc((size_t)H*QT*NSEL,4); float *pqm=calloc((size_t)QT*D,4),*pkm=calloc((size_t)QT*D,4);
  float *sc=calloc(QT,4); uint32_t *idx=calloc(QT,4); size_t pairs=0;
  for(uint32_t h=0;h<H;h++){
    memset(pqm,0,QT*D*4);memset(pkm,0,QT*D*4);
    for(uint32_t r=0;r<S;r++)for(uint32_t d=0;d<D;d++){pqm[(r/64)*D+d]+=bf2f(pq[((size_t)h*S+r)*D+d]);pkm[(r/64)*D+d]+=bf2f(pk[((size_t)h*S+r)*D+d]);}
    for(uint32_t t=0;t<QT;t++){uint32_t c=(t+1)*64>S?S-t*64:64;for(uint32_t d=0;d<D;d++){pqm[t*D+d]/=c;pkm[t*D+d]/=c;}}
    for(uint32_t t=0;t<QT;t++){uint32_t*row=sel+((size_t)h*QT+t)*NSEL;
      if(t<P){row[0]=QT;for(uint32_t u=0;u<QT;u++)row[1+u]=u;pairs+=QT;continue;}
      for(uint32_t u=0;u<QT;u++){float s=0;for(uint32_t d=0;d<D;d++)s+=pqm[t*D+d]*pkm[u*D+d];sc[u]=s*scale;}
      uint32_t m=0;for(uint32_t u=0;u<P;u++)row[1+m++]=u;
      for(uint32_t u=0;u<NV;u++)idx[u]=P+u; gscore=sc; qsort(idx,NV,4,cmpd);
      for(uint32_t j=0;j<K;j++)row[1+m++]=idx[j]; row[0]=m; pairs+=m;}
  }
  id<MTLBuffer> bS=[dev newBufferWithBytes:sel length:(size_t)H*QT*NSEL*4 options:0];
  // dense full list (validates kernel vs MPSGraph)
  uint32_t *full=calloc((size_t)H*QT*NSEL,4);
  for(size_t i=0;i<(size_t)H*QT;i++){full[i*NSEL]=QT;for(uint32_t u=0;u<QT;u++)full[i*NSEL+1+u]=u;}
  id<MTLBuffer> bF=[dev newBufferWithBytes:full length:(size_t)H*QT*NSEL*4 options:0];
  printf("S=%u H=%u tiles=%u prefix=%u video=%u topk=%u  kept pairs %.1f%%\n",S,H,QT,P,NV,K,100.0*pairs/((double)H*QT*QT));
  // --- MPSGraph dense ---
  MPSGraph*gr=[MPSGraph new]; NSArray*shp=@[@1,@(H),@(S),@(D)];
  MPSGraphTensor*tq=[gr placeholderWithShape:shp dataType:MPSDataTypeBFloat16 name:nil],*tk=[gr placeholderWithShape:shp dataType:MPSDataTypeBFloat16 name:nil],*tv=[gr placeholderWithShape:shp dataType:MPSDataTypeBFloat16 name:nil];
  MPSGraphTensor*to=[gr scaledDotProductAttentionWithQueryTensor:tq keyTensor:tk valueTensor:tv scale:scale name:nil];
  MPSGraphTensorData*(^td)(id<MTLBuffer>)=^(id<MTLBuffer> b){return [[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:shp dataType:MPSDataTypeBFloat16];};
  NSDictionary*feeds=@{tq:td(bQ),tk:td(bK),tv:td(bV)},*res=@{to:td(bR)};
  double best=1e9;
  for(int r=0;r<REPS+1;r++){MPSCommandBuffer*cb=[MPSCommandBuffer commandBufferFromCommandQueue:q];
    [gr encodeToCommandBuffer:cb feeds:feeds targetOperations:nil resultsDictionary:res executionDescriptor:nil];
    double w0=CACurrentMediaTime();[cb commit];[cb waitUntilCompleted];double t=CACurrentMediaTime()-w0;if(r>0&&t<best)best=t;}
  printf("MPSGraph dense SDPA wall %.3f ms\n",best*1e3);
  NSString*src=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:getenv("BS_METAL")?:"bs.metal"] encoding:NSUTF8StringEncoding error:nil];
  MTLCompileOptions*o=[MTLCompileOptions new]; o.mathMode=MTLMathModeFast; NSError*e=nil;
  id<MTLLibrary> lib=[dev newLibraryWithSource:src options:o error:&e]; if(!lib){printf("compile fail %s\n",e.localizedDescription.UTF8String);return 1;}
  id<MTLComputePipelineState> p=[dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"bs_r16_k64_s4"] error:&e];
  if(!p){printf("pipeline fail %s\n",e.localizedDescription.UTF8String);return 1;}
  bs_args ar={S,H,scale*1.44269504089f,QT,NSEL};
  uint16_t*pr=bR.contents,*po=bO.contents;
  for(int mode=0;mode<2;mode++){
    id<MTLBuffer> list=mode?bS:bF; double bw=1e9,bt=1e9;
    for(int r=0;r<REPS+1;r++){id<MTLCommandBuffer> cb=[q commandBuffer];id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
      [en setComputePipelineState:p];[en setBuffer:bQ offset:0 atIndex:0];[en setBuffer:bK offset:0 atIndex:1];
      [en setBuffer:bV offset:0 atIndex:2];[en setBuffer:bO offset:0 atIndex:3];[en setBytes:&ar length:sizeof ar atIndex:4];[en setBuffer:list offset:0 atIndex:5];
      [en dispatchThreadgroups:MTLSizeMake(QT,H,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
      [en endEncoding];double w0=CACurrentMediaTime();[cb commit];[cb waitUntilCompleted];double wt=CACurrentMediaTime()-w0;
      if(cb.error){printf("err %s\n",cb.error.localizedDescription.UTF8String);return 1;}
      if(r>0&&wt<bw)bw=wt; double t=cb.GPUEndTime-cb.GPUStartTime; if(r>0&&t<bt)bt=t;}
    double maxe=0;
    if(mode==0){for(size_t i=0;i<n;i++){double d=fabs(bf2f(po[i])-bf2f(pr[i]));if(!(d<=maxe))maxe=isfinite(d)?d:INFINITY;}
      printf("bs kernel, full list   gpu %.3f ms wall %.3f ms  maxabs vs MPSGraph %.4g\n",bt*1e3,bw*1e3,maxe);}
    else{ // CPU double reference with the same mask, heads 0, H/3, H-1, all rows
      uint32_t hs[3]={0,H/3,H-1}; double *pr2=malloc(S*8);
      for(int hi=0;hi<3;hi++){uint32_t h=hs[hi];
        for(uint32_t r=0;r<S;r++){uint32_t*row=sel+((size_t)h*QT+r/64)*NSEL; double m=-1e300,l=0,acc[128]={0};
          for(uint32_t j=0;j<row[0];j++)for(uint32_t c=row[1+j]*64;c<row[1+j]*64+64&&c<S;c++){double s=0;
            for(uint32_t d=0;d<D;d++)s+=(double)bf2f(pq[((size_t)h*S+r)*D+d])*bf2f(pk[((size_t)h*S+c)*D+d]); s*=scale;
            if(s>m){double a=exp(m-s);l*=a;for(int d=0;d<128;d++)acc[d]*=a;m=s;} double w=exp(s-m);l+=w;
            for(uint32_t d=0;d<D;d++)acc[d]+=w*bf2f(pv[((size_t)h*S+c)*D+d]);}
          for(uint32_t d=0;d<D;d++){double dd=fabs(acc[d]/l-bf2f(po[((size_t)h*S+r)*D+d]));if(!(dd<=maxe))maxe=isfinite(dd)?dd:INFINITY;}}}
      printf("bs kernel, top-k list  gpu %.3f ms wall %.3f ms  speedup vs MPSGraph dense %.2fx  maxabs vs CPU same-mask ref (3 heads) %.4g\n",bt*1e3,bw*1e3,best/bw,maxe);}
  }
}return 0;}
