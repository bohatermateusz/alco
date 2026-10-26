package com.hismith.hismithapp.adapter;

import android.graphics.Bitmap;
import android.graphics.Typeface;
import android.text.TextUtils;
import android.util.SparseArray;
import android.view.View;
import android.widget.Checkable;
import android.widget.ImageView;
import android.widget.TextView;
import java.lang.ref.WeakReference;

/* loaded from: classes3.dex */
public class XQuickViewHolder {
    private View itemView;
    private int mLayoutId;
    private SparseArray<WeakReference<View>> mViews;

    public XQuickViewHolder(View view) {
        this(view, -1);
    }

    public XQuickViewHolder(View view, int i) {
        this.itemView = view;
        this.mViews = new SparseArray<>();
        this.mLayoutId = i;
    }

    public XQuickViewHolder displayCircleImage(int i, String str, int i2) {
        return this;
    }

    public XQuickViewHolder displayImage(int i, int i2) {
        return this;
    }

    public XQuickViewHolder displayImage(int i, String str) {
        return this;
    }

    public XQuickViewHolder displayImage(int i, String str, int i2) {
        return this;
    }

    public XQuickViewHolder displayRadiusImage(int i, String str, int i2, int i3) {
        return this;
    }

    public int getLayoutId() {
        return this.mLayoutId;
    }

    public View getView() {
        return this.itemView;
    }

    public <T extends View> T getView(int i) {
        WeakReference<View> weakReference = this.mViews.get(i);
        if (weakReference != null) {
            return (T) weakReference.get();
        }
        T t = (T) this.itemView.findViewById(i);
        if (t != null) {
            this.mViews.put(i, new WeakReference<>(t));
        }
        return t;
    }

    public XQuickViewHolder setBackgroundRes(int i, int i2) {
        getView(i).setBackgroundResource(i2);
        return this;
    }

    public XQuickViewHolder setBold(int i, boolean z) {
        TextView textView = (TextView) getView(i);
        if (textView != null) {
            if (z) {
                textView.setTypeface(Typeface.defaultFromStyle(1));
                return this;
            }
            textView.setTypeface(Typeface.defaultFromStyle(0));
        }
        return this;
    }

    public XQuickViewHolder setBold(TextView textView, boolean z) {
        if (textView != null) {
            if (z) {
                textView.setTypeface(Typeface.defaultFromStyle(1));
                return this;
            }
            textView.setTypeface(Typeface.defaultFromStyle(0));
        }
        return this;
    }

    public XQuickViewHolder setChecked(int i, boolean z) {
        ((Checkable) getView(i)).setChecked(z);
        return this;
    }

    public XQuickViewHolder setDeleteLine(int i, boolean z) {
        TextView textView = (TextView) getView(i);
        if (textView != null) {
            if (z) {
                textView.setPaintFlags(textView.getPaintFlags() | 16);
                return this;
            }
            textView.setPaintFlags(textView.getPaintFlags() & (-17));
        }
        return this;
    }

    public XQuickViewHolder setDeleteLine(TextView textView, boolean z) {
        if (textView != null) {
            if (z) {
                textView.setPaintFlags(textView.getPaintFlags() | 16);
                return this;
            }
            textView.setPaintFlags(textView.getPaintFlags() & (-17));
        }
        return this;
    }

    public XQuickViewHolder setImageBitmap(int i, Bitmap bitmap) {
        ((ImageView) getView(i)).setImageBitmap(bitmap);
        return this;
    }

    public XQuickViewHolder setImageResource(int i, int i2) {
        ((ImageView) getView(i)).setImageResource(i2);
        return this;
    }

    public XQuickViewHolder setOnClickListener(int i, View.OnClickListener onClickListener) {
        View view = getView(i);
        if (view != null) {
            view.setOnClickListener(onClickListener);
        }
        return this;
    }

    public XQuickViewHolder setOnClickListener(View.OnClickListener onClickListener) {
        this.itemView.setOnClickListener(onClickListener);
        return this;
    }

    public XQuickViewHolder setOnLongClickListener(int i, View.OnLongClickListener onLongClickListener) {
        View view = getView(i);
        if (view != null) {
            view.setOnLongClickListener(onLongClickListener);
        }
        return this;
    }

    public XQuickViewHolder setOnLongClickListener(View.OnLongClickListener onLongClickListener) {
        this.itemView.setOnLongClickListener(onLongClickListener);
        return this;
    }

    public XQuickViewHolder setText(int i, CharSequence charSequence) {
        TextView textView = (TextView) getView(i);
        if (textView != null && !TextUtils.isEmpty(charSequence)) {
            textView.setText(charSequence);
        }
        return this;
    }

    public XQuickViewHolder setTextColor(int i, int i2) {
        TextView textView = (TextView) getView(i);
        if (textView != null) {
            textView.setTextColor(i2);
        }
        return this;
    }

    public XQuickViewHolder setUnderLine(int i, boolean z) {
        TextView textView = (TextView) getView(i);
        if (textView != null) {
            if (z) {
                textView.setPaintFlags(textView.getPaintFlags() | 8);
                return this;
            }
            textView.setPaintFlags(textView.getPaintFlags() & (-9));
        }
        return this;
    }

    public XQuickViewHolder setUnderLine(TextView textView, boolean z) {
        if (textView != null) {
            if (z) {
                textView.setPaintFlags(textView.getPaintFlags() | 8);
                return this;
            }
            textView.setPaintFlags(textView.getPaintFlags() & (-9));
        }
        return this;
    }

    public XQuickViewHolder setVisible(int i, int i2) {
        getView(i).setVisibility(i2);
        return this;
    }

    public XQuickViewHolder setVisible(int i, boolean z) {
        View view = getView(i);
        if (z) {
            view.setVisibility(0);
            return this;
        }
        view.setVisibility(8);
        return this;
    }
}
