// Switch scripts the way the UI does and see whether the second run hangs.
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include "../src/core/worker.h"
#include "../src/script/interp.h"
#include "../src/script/parser.h"
#include "../src/core/image_io.h"
#include "../src/app/file_watch.h"
#include <d3d12.h>
using namespace tglab;
using clockt=std::chrono::steady_clock;

static bool Build(const char* path, UiState* ui, Pipeline* pipe,
                  std::vector<Data>* src, std::string* err){
  Image img;
  if(!LoadImageFile("assets/test.png",&img,err)) return false;
  src->clear(); src->push_back(Data{std::move(img)});
  std::vector<SourceImage> names{{"test",0}};
  std::string text;
  if(!ReadTextFile(path,&text)){*err="read failed";return false;}
  Program prog;
  if(!Parse(text,&prog,err)) return false;
  auto r=Interpret(prog,names,ui,pipe);
  if(!r.ok){*err=r.error;return false;}
  return true;
}

int main(){
  ID3D12Device* dev=nullptr;
  D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&dev));
  PipelineWorker w; w.Start(dev);

  // UiState is shared across the switch, exactly as the app does it.
  UiState ui;
  const char* scripts[]={"scripts/hello.tgl","scripts/thresholds.tgl","scripts/hello.tgl"};
  for(const char* s : scripts){
    Pipeline pipe; std::vector<Data> src; std::string err;
    printf("--- %s\n",s);
    if(!Build(s,&ui,&pipe,&src,&err)){printf("    build failed: %s\n",err.c_str());continue;}
    printf("    stages=%zu viewers=%zu controls=%zu\n",
           pipe.Stages().size(),pipe.Viewers().size(),ui.Controls().size());
    w.Submit(std::move(pipe),std::move(src));
    PipelineOutcome out; const auto t0=clockt::now(); bool got=false;
    while(std::chrono::duration<double>(clockt::now()-t0).count()<20.0){
      if(w.TryFetch(&out)){got=true;break;}
      Sleep(5);
    }
    if(!got){printf("    *** TIMED OUT (hang reproduced) ***\n"); w.Stop(); return 1;}
    printf("    ok=%d viewers=%zu err=%s\n",int(out.ok),out.viewers.size(),out.error.c_str());
  }
  w.Stop(); if(dev)dev->Release();
  printf("no hang\n");
  return 0;
}
