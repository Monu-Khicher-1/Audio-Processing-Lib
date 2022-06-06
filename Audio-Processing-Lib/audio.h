#ifndef audio_h__
#define audio_h__
 
typedef struct{
    int label;
    float prob;
}pred_t;

extern pred_t* libaudioAPI(const char* audiofeatures, pred_t* pred);
 
#endif  // audio_h__