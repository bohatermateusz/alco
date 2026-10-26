package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.graphics.Color;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class PlayModeColorAdapter extends RecyclerView.Adapter<ViewHolder> {
    private String[] colorArr = {"#ffffff", "#ff0000", "#ff0072", "#7963ff", "#0bd322", "#00fee9", "#effe00", "#ff9c00"};
    private Context mContext;
    private OnItemClickListener onItemClickListener;

    public interface OnItemClickListener {
        void onItemClick(String str);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private FrameLayout mLayoutColor;
        private View mViewColor;

        public ViewHolder(View view) {
            super(view);
            this.mViewColor = view.findViewById(R.id.view_color);
            FrameLayout frameLayout = (FrameLayout) view.findViewById(R.id.layout_color);
            this.mLayoutColor = frameLayout;
            ViewGroup.LayoutParams layoutParams = frameLayout.getLayoutParams();
            int screenWidth = (int) (MeasureUtil.getScreenWidth(PlayModeColorAdapter.this.mContext) * 0.064d);
            layoutParams.width = screenWidth;
            layoutParams.height = screenWidth;
            this.mLayoutColor.setLayoutParams(layoutParams);
        }
    }

    public PlayModeColorAdapter(Context context) {
        this.mContext = context;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.colorArr.length;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        viewHolder.mViewColor.setBackgroundColor(Color.parseColor(this.colorArr[i]));
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlayModeColorAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (PlayModeColorAdapter.this.onItemClickListener != null) {
                    PlayModeColorAdapter.this.onItemClickListener.onItemClick(PlayModeColorAdapter.this.colorArr[i]);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.item_play_mode_color, viewGroup, false));
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
