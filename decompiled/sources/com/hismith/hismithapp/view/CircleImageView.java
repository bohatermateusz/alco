package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.BitmapShader;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.Shader;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.util.Log;
import androidx.appcompat.widget.AppCompatImageView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.load.DataSource;
import com.bumptech.glide.load.engine.DiskCacheStrategy;
import com.bumptech.glide.load.engine.GlideException;
import com.bumptech.glide.load.resource.gif.GifDrawable;
import com.bumptech.glide.request.RequestListener;
import com.bumptech.glide.request.target.CustomTarget;
import com.bumptech.glide.request.target.Target;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.R;
import java.util.concurrent.atomic.AtomicBoolean;

/* loaded from: classes3.dex */
public class CircleImageView extends AppCompatImageView {
    private Paint borderPaint;
    private Bitmap currentBitmap;
    private GifDrawable currentGifDrawable;
    private String gradeIcon;
    private Bitmap gradeIconBitmap;
    private float height;
    private AtomicBoolean isGif;
    private int isKing;
    private int isOnline;
    private Paint markPaint;
    private int markRadius;
    private Matrix matrix;
    private Paint paint;
    private float radius;
    private float width;

    public CircleImageView(Context context) {
        this(context, null);
    }

    public CircleImageView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public CircleImageView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.isKing = 0;
        this.gradeIcon = "";
        this.isOnline = -1;
        this.isGif = new AtomicBoolean(false);
        Paint paint = new Paint();
        this.paint = paint;
        paint.setAntiAlias(true);
        Paint paint2 = new Paint();
        this.markPaint = paint2;
        paint2.setAntiAlias(true);
        this.markPaint.setColor(-1);
        this.markPaint.setStyle(Paint.Style.FILL);
        Paint paint3 = new Paint();
        this.borderPaint = paint3;
        paint3.setAntiAlias(true);
        this.borderPaint.setStyle(Paint.Style.STROKE);
        this.borderPaint.setStrokeWidth(1.0f);
        this.borderPaint.setColor(Color.parseColor("#999999"));
        this.matrix = new Matrix();
    }

    private void clearResources() {
        GifDrawable gifDrawable = this.currentGifDrawable;
        if (gifDrawable != null) {
            gifDrawable.stop();
            this.currentGifDrawable = null;
        }
        this.currentBitmap = null;
        this.isGif.set(false);
    }

    private void drawBitmapAsCircle(Canvas canvas, Drawable drawable) {
        try {
            Bitmap bitmap = this.currentBitmap;
            if (bitmap == null) {
                bitmap = drawable instanceof BitmapDrawable ? ((BitmapDrawable) drawable).getBitmap() : drawableToBitmap(drawable);
            }
            if (bitmap == null) {
                return;
            }
            if (this.isKing == 0 && this.isOnline == 0) {
                ColorMatrix colorMatrix = new ColorMatrix();
                colorMatrix.setSaturation(0.0f);
                this.paint.setColorFilter(new ColorMatrixColorFilter(colorMatrix));
            } else {
                ColorMatrix colorMatrix2 = new ColorMatrix();
                colorMatrix2.setSaturation(1.0f);
                this.paint.setColorFilter(new ColorMatrixColorFilter(colorMatrix2));
            }
            this.paint.setShader(initBitmapShader(bitmap));
            canvas.drawCircle(this.width / 2.0f, this.height / 2.0f, this.radius, this.paint);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void drawGifAsCircle(Canvas canvas) {
        if (this.currentGifDrawable == null) {
            return;
        }
        try {
            Bitmap createBitmap = Bitmap.createBitmap((int) this.width, (int) this.height, Bitmap.Config.ARGB_8888);
            Canvas canvas2 = new Canvas(createBitmap);
            if (this.isKing == 0 && this.isOnline == 0) {
                ColorMatrix colorMatrix = new ColorMatrix();
                colorMatrix.setSaturation(0.0f);
                this.paint.setColorFilter(new ColorMatrixColorFilter(colorMatrix));
            } else {
                ColorMatrix colorMatrix2 = new ColorMatrix();
                colorMatrix2.setSaturation(1.0f);
                this.paint.setColorFilter(new ColorMatrixColorFilter(colorMatrix2));
            }
            this.currentGifDrawable.setBounds(0, 0, (int) this.width, (int) this.height);
            this.currentGifDrawable.draw(canvas2);
            canvas.drawBitmap(getCroppedRoundBitmap(createBitmap, (int) this.radius), 0.0f, 0.0f, this.paint);
            createBitmap.recycle();
        } catch (Exception e) {
            e.printStackTrace();
            this.currentGifDrawable.setBounds(0, 0, (int) this.width, (int) this.height);
            this.currentGifDrawable.draw(canvas);
        }
    }

    private void drawMark(Canvas canvas) {
        if (this.isKing == 0 && this.isOnline == 0) {
            Bitmap croppedRoundBitmap = getCroppedRoundBitmap(BitmapFactory.decodeResource(getResources(), R.mipmap.ic_unonline), this.markRadius);
            float f = this.width;
            int i = this.markRadius;
            canvas.drawBitmap(croppedRoundBitmap, f - (i * 2), f - (i * 2), (Paint) null);
            return;
        }
        if (TextUtils.isEmpty(this.gradeIcon)) {
            return;
        }
        float f2 = this.width;
        int i2 = this.markRadius;
        canvas.drawCircle(f2 - i2, this.height - i2, i2, this.markPaint);
        float f3 = this.width;
        int i3 = this.markRadius;
        canvas.drawCircle(f3 - i3, this.height - i3, i3 - (this.borderPaint.getStrokeWidth() / 2.0f), this.borderPaint);
        Bitmap bitmap = this.gradeIconBitmap;
        if (bitmap != null) {
            Bitmap croppedRoundBitmap2 = getCroppedRoundBitmap(bitmap, this.markRadius);
            float f4 = this.width;
            int i4 = this.markRadius;
            canvas.drawBitmap(croppedRoundBitmap2, f4 - (i4 * 2), f4 - (i4 * 2), (Paint) null);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public Bitmap drawableToBitmap(Drawable drawable) {
        if (drawable instanceof BitmapDrawable) {
            return ((BitmapDrawable) drawable).getBitmap();
        }
        int intrinsicWidth = drawable.getIntrinsicWidth();
        int intrinsicHeight = drawable.getIntrinsicHeight();
        if (intrinsicWidth <= 0 || intrinsicHeight <= 0) {
            intrinsicWidth = 100;
            intrinsicHeight = 100;
        }
        Bitmap createBitmap = Bitmap.createBitmap(intrinsicWidth, intrinsicHeight, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(createBitmap);
        drawable.setBounds(0, 0, canvas.getWidth(), canvas.getHeight());
        drawable.draw(canvas);
        return createBitmap;
    }

    private BitmapShader initBitmapShader(Bitmap bitmap) {
        BitmapShader bitmapShader = new BitmapShader(bitmap, Shader.TileMode.CLAMP, Shader.TileMode.CLAMP);
        float max = Math.max(this.width / bitmap.getWidth(), this.height / bitmap.getHeight());
        this.matrix.setScale(max, max);
        bitmapShader.setLocalMatrix(this.matrix);
        return bitmapShader;
    }

    public Bitmap getCroppedRoundBitmap(Bitmap bitmap, int i) {
        if (bitmap == null) {
            return null;
        }
        int i2 = i * 2;
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Bitmap createBitmap = height > width ? Bitmap.createBitmap(bitmap, 0, (height - width) / 2, width, width) : height < width ? Bitmap.createBitmap(bitmap, (width - height) / 2, 0, height, height) : bitmap;
        Bitmap createScaledBitmap = (createBitmap.getWidth() == i2 && createBitmap.getHeight() == i2) ? createBitmap : Bitmap.createScaledBitmap(createBitmap, i2, i2, true);
        Bitmap createBitmap2 = Bitmap.createBitmap(createScaledBitmap.getWidth(), createScaledBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(createBitmap2);
        Paint paint = new Paint();
        Rect rect = new Rect(0, 0, createScaledBitmap.getWidth(), createScaledBitmap.getHeight());
        paint.setAntiAlias(true);
        paint.setFilterBitmap(true);
        paint.setDither(true);
        canvas.drawARGB(0, 0, 0, 0);
        canvas.drawCircle(createScaledBitmap.getWidth() / 2, createScaledBitmap.getHeight() / 2, createScaledBitmap.getWidth() / 2, paint);
        paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC_IN));
        canvas.drawBitmap(createScaledBitmap, rect, rect, paint);
        if (createBitmap != bitmap) {
            createBitmap.recycle();
        }
        return createBitmap2;
    }

    public String getGradeIcon() {
        return this.gradeIcon;
    }

    public int getIsOnline() {
        return this.isOnline;
    }

    public int isKing() {
        return this.isKing;
    }

    public void loadImageWithGlide(String str, Context context) {
        if (context == null) {
            return;
        }
        clearResources();
        Glide.with(context).load(str).diskCacheStrategy(DiskCacheStrategy.ALL).listener(new RequestListener<Drawable>() { // from class: com.hismith.hismithapp.view.CircleImageView.2
            @Override // com.bumptech.glide.request.RequestListener
            public boolean onLoadFailed(GlideException glideException, Object obj, Target<Drawable> target, boolean z) {
                Log.e("CircleImageView", "图片加载失败: " + (glideException != null ? glideException.getMessage() : "未知错误"));
                CircleImageView.this.isGif.set(false);
                return false;
            }

            @Override // com.bumptech.glide.request.RequestListener
            public boolean onResourceReady(Drawable drawable, Object obj, Target<Drawable> target, DataSource dataSource, boolean z) {
                boolean z2 = drawable instanceof GifDrawable;
                CircleImageView circleImageView = CircleImageView.this;
                if (z2) {
                    circleImageView.isGif.set(true);
                    CircleImageView.this.currentGifDrawable = (GifDrawable) drawable;
                    CircleImageView.this.currentGifDrawable.setLoopCount(-1);
                    CircleImageView.this.currentGifDrawable.start();
                    CircleImageView circleImageView2 = CircleImageView.this;
                    circleImageView2.setImageDrawable(circleImageView2.currentGifDrawable);
                } else {
                    circleImageView.isGif.set(false);
                    CircleImageView circleImageView3 = CircleImageView.this;
                    circleImageView3.currentBitmap = circleImageView3.drawableToBitmap(drawable);
                    CircleImageView.this.setImageDrawable(drawable);
                }
                return false;
            }
        }).into(this);
    }

    public void loadImageWithGlide(String str, Context context, int i) {
        if (context == null) {
            return;
        }
        clearResources();
        Glide.with(context).load(str).placeholder(i).diskCacheStrategy(DiskCacheStrategy.ALL).listener(new RequestListener<Drawable>() { // from class: com.hismith.hismithapp.view.CircleImageView.3
            @Override // com.bumptech.glide.request.RequestListener
            public boolean onLoadFailed(GlideException glideException, Object obj, Target<Drawable> target, boolean z) {
                Log.e("CircleImageView", "图片加载失败: " + (glideException != null ? glideException.getMessage() : "未知错误"));
                CircleImageView.this.isGif.set(false);
                return false;
            }

            @Override // com.bumptech.glide.request.RequestListener
            public boolean onResourceReady(Drawable drawable, Object obj, Target<Drawable> target, DataSource dataSource, boolean z) {
                boolean z2 = drawable instanceof GifDrawable;
                CircleImageView circleImageView = CircleImageView.this;
                if (z2) {
                    circleImageView.isGif.set(true);
                    CircleImageView.this.currentGifDrawable = (GifDrawable) drawable;
                    CircleImageView.this.currentGifDrawable.setLoopCount(-1);
                    CircleImageView.this.currentGifDrawable.start();
                    CircleImageView circleImageView2 = CircleImageView.this;
                    circleImageView2.setImageDrawable(circleImageView2.currentGifDrawable);
                } else {
                    circleImageView.isGif.set(false);
                    CircleImageView circleImageView3 = CircleImageView.this;
                    circleImageView3.currentBitmap = circleImageView3.drawableToBitmap(drawable);
                    CircleImageView.this.setImageDrawable(drawable);
                }
                return false;
            }
        }).into(this);
    }

    @Override // android.widget.ImageView, android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        clearResources();
    }

    @Override // android.widget.ImageView, android.view.View
    protected void onDraw(Canvas canvas) {
        if (!this.isGif.get() || this.currentGifDrawable == null) {
            Drawable drawable = getDrawable();
            if (drawable == null) {
                return;
            } else {
                drawBitmapAsCircle(canvas, drawable);
            }
        } else {
            drawGifAsCircle(canvas);
        }
        drawMark(canvas);
    }

    @Override // android.widget.ImageView, android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        this.width = getMeasuredWidth();
        float measuredHeight = getMeasuredHeight();
        this.height = measuredHeight;
        float min = Math.min(this.width, measuredHeight) / 2.0f;
        this.radius = min;
        this.markRadius = (int) (min / 4.0f);
    }

    public void pauseGif() {
        GifDrawable gifDrawable = this.currentGifDrawable;
        if (gifDrawable == null || !gifDrawable.isRunning()) {
            return;
        }
        this.currentGifDrawable.stop();
    }

    public void resumeGif() {
        GifDrawable gifDrawable = this.currentGifDrawable;
        if (gifDrawable == null || gifDrawable.isRunning()) {
            return;
        }
        this.currentGifDrawable.start();
    }

    public void setGradeIcon(String str) {
        this.gradeIcon = str;
    }

    public void setIsOnline(int i) {
        this.isOnline = i;
        invalidate();
    }

    public void setKing(int i, Context context, String str) {
        this.isKing = i;
        this.gradeIcon = str;
        Glide.with(context).asBitmap().load(str).into((RequestBuilder<Bitmap>) new CustomTarget<Bitmap>() { // from class: com.hismith.hismithapp.view.CircleImageView.1
            @Override // com.bumptech.glide.request.target.Target
            public void onLoadCleared(Drawable drawable) {
            }

            public void onResourceReady(Bitmap bitmap, Transition<? super Bitmap> transition) {
                try {
                    CircleImageView.this.gradeIconBitmap = bitmap;
                    CircleImageView.this.invalidate();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            @Override // com.bumptech.glide.request.target.Target
            public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                onResourceReady((Bitmap) obj, (Transition<? super Bitmap>) transition);
            }
        });
        invalidate();
    }
}
