package com.hismith.hismithapp.usb;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbEndpoint;
import android.hardware.usb.UsbInterface;
import android.hardware.usb.UsbManager;
import android.net.Uri;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.view.SurfaceView;
import android.widget.Toast;
import androidx.core.view.ViewCompat;
import com.hismith.hismithapp.utils.LogUtil2;
import com.luck.picture.lib.config.PictureMimeType;
import java.io.IOException;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

/* loaded from: classes3.dex */
public class UVCCameraHandler {
    private static final String TAG = "UVCCameraHandler";
    private UsbDeviceConnection connection;
    private Context context;
    private boolean isPreviewing = false;
    private PreviewThread previewThread;
    private SurfaceView surfaceView;
    private UsbDevice usbDevice;
    private UsbInterface usbInterface;
    private UsbManager usbManager;
    private UsbEndpoint videoEndpoint;

    private class CaptureThread extends Thread {
        private ImageCaptureCallback callback;

        public CaptureThread(ImageCaptureCallback imageCaptureCallback) {
            this.callback = imageCaptureCallback;
        }

        private int clamp(int i) {
            return Math.max(0, Math.min(255, i));
        }

        private Bitmap decodeYUYV(byte[] bArr, int i, int i2) {
            int i3 = i * i2;
            int[] iArr = new int[i3];
            int i4 = 0;
            int i5 = 0;
            while (i4 < bArr.length - 3 && i5 < i3) {
                int i6 = bArr[i4] & 255;
                int i7 = bArr[i4 + 1] & 255;
                int i8 = bArr[i4 + 2] & 255;
                double d = i6;
                double d2 = (bArr[i4 + 3] & 255) - 128;
                double d3 = 1.402d * d2;
                int i9 = i4;
                int i10 = i5;
                int clamp = clamp((int) (d + d3));
                double d4 = i7 - 128;
                double d5 = 0.344d * d4;
                double d6 = d2 * 0.714d;
                int clamp2 = clamp((int) ((d - d5) - d6));
                double d7 = 1.772d * d4;
                int clamp3 = clamp((int) (d + d7));
                double d8 = i8;
                int clamp4 = clamp((int) (d8 + d3));
                int clamp5 = clamp((int) ((d8 - d5) - d6));
                int clamp6 = clamp((int) (d8 + d7));
                int i11 = i10 + 1;
                iArr[i10] = (clamp2 << 8) | (clamp << 16) | ViewCompat.MEASURED_STATE_MASK | clamp3;
                if (i11 < i3) {
                    i5 = i10 + 2;
                    iArr[i11] = (clamp4 << 16) | ViewCompat.MEASURED_STATE_MASK | (clamp5 << 8) | clamp6;
                } else {
                    i5 = i11;
                }
                i4 = i9 + 4;
            }
            return Bitmap.createBitmap(iArr, i, i2, Bitmap.Config.ARGB_8888);
        }

        private Bitmap processFrameForCapture(byte[] bArr, int i) {
            Bitmap decodeByteArray = BitmapFactory.decodeByteArray(bArr, 0, i);
            return decodeByteArray == null ? decodeYUYV(bArr, 640, 480) : decodeByteArray;
        }

        private void saveImageToGallery(Bitmap bitmap) {
            String str = "IMG_" + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date()) + PictureMimeType.JPG;
            ContentValues contentValues = new ContentValues();
            contentValues.put("_display_name", str);
            contentValues.put("mime_type", "image/jpeg");
            contentValues.put("relative_path", Environment.DIRECTORY_PICTURES + "/USBCamera");
            ContentResolver contentResolver = UVCCameraHandler.this.context.getContentResolver();
            Uri insert = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues);
            if (insert == null) {
                showToast("Failed to save photo: unable to create content URI");
                return;
            }
            try {
                OutputStream openOutputStream = contentResolver.openOutputStream(insert);
                if (openOutputStream != null) {
                    try {
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, openOutputStream);
                        showToast("Photo saved to gallery");
                    } finally {
                    }
                }
                if (openOutputStream != null) {
                    openOutputStream.close();
                }
            } catch (IOException e) {
                LogUtil2.e("Failed to save image", e);
                showToast("Failed to save photo: " + e.getMessage());
            }
        }

        private void showToast(final String str) {
            new Handler(Looper.getMainLooper()).post(new Runnable() { // from class: com.hismith.hismithapp.usb.UVCCameraHandler.CaptureThread.3
                @Override // java.lang.Runnable
                public void run() {
                    Toast.makeText(UVCCameraHandler.this.context, str, 0).show();
                }
            });
        }

        @Override // java.lang.Thread, java.lang.Runnable
        public void run() {
            try {
                int maxPacketSize = UVCCameraHandler.this.videoEndpoint.getMaxPacketSize() * 100;
                byte[] bArr = new byte[maxPacketSize];
                int bulkTransfer = UVCCameraHandler.this.connection.bulkTransfer(UVCCameraHandler.this.videoEndpoint, bArr, maxPacketSize, 100);
                if (bulkTransfer <= 0) {
                    throw new Exception("Failed to read data from camera");
                }
                final Bitmap processFrameForCapture = processFrameForCapture(bArr, bulkTransfer);
                if (processFrameForCapture == null) {
                    throw new Exception("Frame processing failed");
                }
                new Handler(Looper.getMainLooper()).post(new Runnable() { // from class: com.hismith.hismithapp.usb.UVCCameraHandler.CaptureThread.1
                    @Override // java.lang.Runnable
                    public void run() {
                        CaptureThread.this.callback.onImageCaptured(processFrameForCapture);
                    }
                });
                saveImageToGallery(processFrameForCapture);
            } catch (Exception e) {
                final String message = e.getMessage();
                new Handler(Looper.getMainLooper()).post(new Runnable() { // from class: com.hismith.hismithapp.usb.UVCCameraHandler.CaptureThread.2
                    @Override // java.lang.Runnable
                    public void run() {
                        CaptureThread.this.callback.onError(message);
                    }
                });
            }
        }
    }

    public interface ImageCaptureCallback {
        void onError(String str);

        void onImageCaptured(Bitmap bitmap);
    }

    private class PreviewThread extends Thread {
        private PreviewThread() {
        }

        private Bitmap processFrame(byte[] bArr, int i) {
            try {
                return BitmapFactory.decodeByteArray(bArr, 0, i);
            } catch (Exception e) {
                LogUtil2.w("Failed to decode frame " + e.getMessage());
                return null;
            }
        }

        private void processVideoData(byte[] bArr, int i) {
            Bitmap processFrame = processFrame(bArr, i);
            if (processFrame != null) {
                renderToSurfaceView(processFrame);
            }
        }

        private void renderToSurfaceView(final Bitmap bitmap) {
            if (bitmap == null) {
                return;
            }
            new Handler(Looper.getMainLooper()).post(new Runnable() { // from class: com.hismith.hismithapp.usb.UVCCameraHandler.PreviewThread.1
                @Override // java.lang.Runnable
                public void run() {
                    try {
                        Canvas lockCanvas = UVCCameraHandler.this.surfaceView.getHolder().lockCanvas();
                        if (lockCanvas != null) {
                            lockCanvas.drawBitmap(bitmap, (Rect) null, UVCCameraHandler.this.calculateDestinationRect(bitmap.getWidth(), bitmap.getHeight(), UVCCameraHandler.this.surfaceView.getWidth(), UVCCameraHandler.this.surfaceView.getHeight()), (Paint) null);
                            UVCCameraHandler.this.surfaceView.getHolder().unlockCanvasAndPost(lockCanvas);
                        }
                    } catch (Exception e) {
                        LogUtil2.w("Failed to render to surface " + e.getMessage());
                    }
                }
            });
        }

        @Override // java.lang.Thread, java.lang.Runnable
        public void run() {
            int maxPacketSize = UVCCameraHandler.this.videoEndpoint.getMaxPacketSize() * 100;
            byte[] bArr = new byte[maxPacketSize];
            while (UVCCameraHandler.this.isPreviewing) {
                int bulkTransfer = UVCCameraHandler.this.connection.bulkTransfer(UVCCameraHandler.this.videoEndpoint, bArr, maxPacketSize, 100);
                if (bulkTransfer > 0) {
                    processVideoData(bArr, bulkTransfer);
                } else if (bulkTransfer == 0) {
                    try {
                        Thread.sleep(10L);
                    } catch (InterruptedException unused) {
                        return;
                    }
                } else {
                    LogUtil2.w("Bulk transfer error: " + bulkTransfer);
                }
            }
        }
    }

    public UVCCameraHandler(Context context, UsbManager usbManager, UsbDevice usbDevice, SurfaceView surfaceView) {
        this.context = context;
        this.usbManager = usbManager;
        this.usbDevice = usbDevice;
        this.surfaceView = surfaceView;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public Rect calculateDestinationRect(int i, int i2, int i3, int i4) {
        int i5;
        float f = i / i2;
        float f2 = i3;
        float f3 = i4;
        int i6 = 0;
        if (f > f2 / f3) {
            int i7 = (int) (f2 / f);
            int i8 = (i4 - i7) / 2;
            i4 = i7;
            i5 = i8;
        } else {
            int i9 = (int) (f3 * f);
            int i10 = (i3 - i9) / 2;
            i3 = i9;
            i5 = 0;
            i6 = i10;
        }
        return new Rect(i6, i5, i3 + i6, i4 + i5);
    }

    public void captureImage(ImageCaptureCallback imageCaptureCallback) {
        if (this.isPreviewing) {
            new CaptureThread(imageCaptureCallback).start();
        } else {
            imageCaptureCallback.onError("Preview not started");
        }
    }

    public boolean initializeCamera() {
        LogUtil2.e("Initializing camera...");
        UsbDeviceConnection openDevice = this.usbManager.openDevice(this.usbDevice);
        this.connection = openDevice;
        if (openDevice == null) {
            LogUtil2.e("Failed to open USB device");
            return false;
        }
        int i = 0;
        while (true) {
            if (i >= this.usbDevice.getInterfaceCount()) {
                break;
            }
            UsbInterface usbInterface = this.usbDevice.getInterface(i);
            if (usbInterface.getInterfaceClass() == 14) {
                this.usbInterface = usbInterface;
                break;
            }
            i++;
        }
        UsbInterface usbInterface2 = this.usbInterface;
        if (usbInterface2 == null) {
            LogUtil2.e("No video interface found");
            this.connection.close();
            return false;
        }
        if (!this.connection.claimInterface(usbInterface2, true)) {
            LogUtil2.e("Failed to claim interface");
            this.connection.close();
            return false;
        }
        int i2 = 0;
        while (true) {
            if (i2 >= this.usbInterface.getEndpointCount()) {
                break;
            }
            UsbEndpoint endpoint = this.usbInterface.getEndpoint(i2);
            if (endpoint.getDirection() == 128) {
                this.videoEndpoint = endpoint;
                break;
            }
            i2++;
        }
        if (this.videoEndpoint != null) {
            LogUtil2.e("Camera initialized successfully");
            return true;
        }
        LogUtil2.e("No video endpoint found");
        this.connection.releaseInterface(this.usbInterface);
        this.connection.close();
        return false;
    }

    public boolean isPreviewing() {
        return this.isPreviewing;
    }

    public void release() {
        stopPreview();
        UsbDeviceConnection usbDeviceConnection = this.connection;
        if (usbDeviceConnection != null) {
            UsbInterface usbInterface = this.usbInterface;
            if (usbInterface != null) {
                usbDeviceConnection.releaseInterface(usbInterface);
            }
            this.connection.close();
            this.connection = null;
        }
        LogUtil2.e("Camera resources released");
    }

    public void startPreview() {
        if (this.isPreviewing) {
            LogUtil2.w("Preview is already running");
            return;
        }
        if (this.videoEndpoint == null) {
            LogUtil2.e("Cannot start preview - camera not initialized");
            return;
        }
        PreviewThread previewThread = new PreviewThread();
        this.previewThread = previewThread;
        previewThread.start();
        this.isPreviewing = true;
        LogUtil2.e("Preview started");
    }

    public void stopPreview() {
        this.isPreviewing = false;
        PreviewThread previewThread = this.previewThread;
        if (previewThread != null) {
            try {
                previewThread.join(1000L);
            } catch (InterruptedException e) {
                LogUtil2.w("Preview thread interrupted " + e.getMessage());
            }
            this.previewThread = null;
        }
        LogUtil2.e("Preview stopped");
    }
}
