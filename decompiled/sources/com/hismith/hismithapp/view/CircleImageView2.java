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
import androidx.appcompat.widget.AppCompatImageView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.CustomTarget;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class CircleImageView2 extends AppCompatImageView {
    private Paint borderPaint;
    private String gradeIcon;
    private Bitmap gradeIconBitmap;
    private float height;
    private int isKing;
    private int isOnline;
    private Paint markPaint;
    private int markRadius;
    private Matrix matrix;
    private Paint paint;
    private float radius;
    private float width;

    public CircleImageView2(Context context) {
        this(context, null);
    }

    public CircleImageView2(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public CircleImageView2(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.isKing = 0;
        this.gradeIcon = "";
        this.isOnline = -1;
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

    private BitmapShader initBitmapShader(BitmapDrawable bitmapDrawable) {
        BitmapShader bitmapShader = new BitmapShader(bitmapDrawable.getBitmap(), Shader.TileMode.CLAMP, Shader.TileMode.CLAMP);
        float max = Math.max(this.width / r4.getWidth(), this.height / r4.getHeight());
        this.matrix.setScale(max, max);
        bitmapShader.setLocalMatrix(this.matrix);
        return bitmapShader;
    }

    public Bitmap getCroppedRoundBitmap(Bitmap bitmap, int i) {
        int i2 = i * 2;
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        if (height > width) {
            bitmap = Bitmap.createBitmap(bitmap, 0, (height - width) / 2, width, width);
        } else if (height < width) {
            bitmap = Bitmap.createBitmap(bitmap, (width - height) / 2, 0, height, height);
        }
        if (bitmap.getWidth() != i2 || bitmap.getHeight() != i2) {
            bitmap = Bitmap.createScaledBitmap(bitmap, i2, i2, true);
        }
        Bitmap createBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(createBitmap);
        Paint paint = new Paint();
        Rect rect = new Rect(0, 0, bitmap.getWidth(), bitmap.getHeight());
        paint.setAntiAlias(true);
        paint.setFilterBitmap(true);
        paint.setDither(true);
        canvas.drawARGB(0, 0, 0, 0);
        canvas.drawCircle(bitmap.getWidth() / 2, bitmap.getHeight() / 2, bitmap.getWidth() / 2, paint);
        paint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.SRC_IN));
        canvas.drawBitmap(bitmap, rect, rect, paint);
        return createBitmap;
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

    @Override // android.widget.ImageView, android.view.View
    protected void onDraw(Canvas canvas) {
        Drawable drawable = getDrawable();
        if (drawable == null) {
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
        this.paint.setShader(initBitmapShader((BitmapDrawable) drawable));
        canvas.drawCircle(this.width / 2.0f, this.height / 2.0f, this.radius, this.paint);
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
        Glide.with(context).asBitmap().load(str).into((RequestBuilder<Bitmap>) new CustomTarget<Bitmap>() { // from class: com.hismith.hismithapp.view.CircleImageView2.1
            @Override // com.bumptech.glide.request.target.Target
            public void onLoadCleared(Drawable drawable) {
            }

            public void onResourceReady(Bitmap bitmap, Transition<? super Bitmap> transition) {
                try {
                    CircleImageView2.this.gradeIconBitmap = bitmap;
                    CircleImageView2.this.invalidate();
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
