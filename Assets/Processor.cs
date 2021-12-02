using UnityEngine;
using UnityEngine.UI;
using Unity.Barracuda;

/**
 * Gets camera frames from a webcam, runs the human segmentation inference and processes image the inference result.
 * Fills the human segment with a background image.
 */
public class Processor : MonoBehaviour
{
    /**
     * RawImage component to display the output of the image processing.
     */
    [SerializeField]
    private RawImage resultImage;

    /**
      * The model to be used for the image processing with Barracuda.
      */
    [SerializeField]
    private NNModel nnModel;

    /**
      * Compute shader for the preprocess. Convert webcame texture to compute buffer.
      */
    [SerializeField]
    private ComputeShader preprocessShader;

    /**
      * Compute shader to combine the current camera image with the previous result.
      */
    [SerializeField]
    private ComputeShader combineShader;

    
    /**
     * Camera resolvetion width.
     */
    [SerializeField]
    private int resolutionX = 1920;

    /**
     * Camera resolvetion height.
     */
    [SerializeField]
    private int resolutionY = 1080;

    /**
      * The height and the width of input image for ML inference.
      */
    const int ML_INPUT_SIZE = 256; 
    /**
     * The size of the ML input channels.
     */
    const int ML_IN_CH = 3;
    /**
      * The size of the ML output channels.
      */
    const int ML_OUT_CH = 1;

    /**
      * Worker instance to run Barracuda inference.
      */
    IWorker woker;

    /**
     * Compute buffer of the ML input.
     */
    private ComputeBuffer mlInputBuffer;

    /**
      * Texture to store a web camera video frame.
      */
    private WebCamTexture cameraTexture;

    /**
      * Texture to store the human segment mask.
      */
    private RenderTexture segmentMask;

    /**
      * No.1
      * Render texture to store the current result and the previous result.
      */
    private RenderTexture resultRenderTexture1;

    /**
      * No.2 
      * Render texture to store the current result and the previous result.
      */
    private RenderTexture resultRenderTexture2;

    /**
     * Count of the video frame.
     */
    private int frameCount = 0;

    void Start()
    {
        cameraTexture = new WebCamTexture("", resolutionX, resolutionY);
        cameraTexture.Play();
        mlInputBuffer = new ComputeBuffer(ML_INPUT_SIZE * ML_INPUT_SIZE * ML_IN_CH, sizeof(float));

        var model = ModelLoader.Load(nnModel);
        woker = model.CreateWorker();

        segmentMask = new RenderTexture(resolutionX, resolutionY, 0, RenderTextureFormat.ARGB32);
        segmentMask.enableRandomWrite = true;
        segmentMask.Create();

        resultRenderTexture1 = new RenderTexture(resolutionX, resolutionY, 1, RenderTextureFormat.ARGBFloat);
        resultRenderTexture1.enableRandomWrite = true;
        resultRenderTexture1.Create();

        resultRenderTexture2 = new RenderTexture(resolutionX, resolutionY, 0, RenderTextureFormat.ARGBFloat);
        resultRenderTexture2.enableRandomWrite = true;
        resultRenderTexture2.Create();
    }

    void Update()
    {
        GenerateSegment(cameraTexture);
        // Counts the frame number and change render texture to be used as the result and the previous result.
        if (frameCount % 2 == 0)
        {
            Combine(cameraTexture, resultRenderTexture1, segmentMask, resultRenderTexture2);
            resultImage.texture = resultRenderTexture1;
        }
        else
        {
            Combine(cameraTexture, resultRenderTexture2, segmentMask, resultRenderTexture1);
            resultImage.texture = resultRenderTexture2;
        }
        frameCount += 1;
    }

    public void Dispose() {
        mlInputBuffer?.Dispose();
        woker?.Dispose();
        resultRenderTexture1?.Release();
        resultRenderTexture2?.Release();
        segmentMask?.Release();
    }

    /**
      * Generates a segment mask texture from the input image.
      * @param cameraTexture The input camera image.
      */
    public void GenerateSegment(Texture cameraTexture) {
        // Convert the camera image to compute buffer.
        preprocessShader.SetTexture(0, "Input", cameraTexture);
        preprocessShader.SetBuffer(0, "Result", mlInputBuffer);
        preprocessShader.Dispatch(0, ML_INPUT_SIZE / 8, ML_INPUT_SIZE / 8, 1);
        // Creates a input tesor with the buffer and execute the model.
        var inputTensor = new Tensor(1, ML_INPUT_SIZE, ML_INPUT_SIZE, ML_IN_CH, mlInputBuffer);
        woker.Execute(inputTensor);
        inputTensor.Dispose();
        // Copy output to the mask texture.
        var shape = new TensorShape(1, ML_INPUT_SIZE, ML_INPUT_SIZE, ML_OUT_CH);
        var tmpOutput = RenderTexture.GetTemporary(ML_INPUT_SIZE, ML_INPUT_SIZE, 0, RenderTextureFormat.ARGB32);
        var tensor = woker.PeekOutput("activation_10").Reshape(shape);
        tensor.ToRenderTexture(tmpOutput);
        tensor.Dispose();
        Graphics.Blit(tmpOutput, segmentMask);
        RenderTexture.ReleaseTemporary(tmpOutput);
    }

    /**
      * Combine the current camera image with the previous result.
      * @param current Current camera image
      * @param previous Previous result
      * @param mask Segmentation mask texture
      * @param output Output texture
      */
    public void Combine(Texture current, Texture previous, Texture mask, Texture output) {
        var kernel = combineShader.FindKernel("CSMain");
        combineShader.SetTexture(kernel, "Current", current);
        combineShader.SetTexture(kernel, "Previous", previous);
        combineShader.SetTexture(kernel, "Output", output);
        combineShader.SetTexture(kernel, "Mask", mask);
        combineShader.Dispatch(
            kernel,
            current.width / 8,
            current.height / 8,
            1
        );
    }
}
