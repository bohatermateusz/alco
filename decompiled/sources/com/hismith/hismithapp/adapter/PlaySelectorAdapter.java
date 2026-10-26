package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.PlaySelectorModel;
import com.hismith.hismithapp.utils.LanguageUtil;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySelectorAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Context mContext;
    private List<PlaySelectorModel> mList;
    private OnItemClickListener onItemClickListener;
    private int itemHeight = 0;
    private int rowSpacingDip = 0;

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView img_play_icon;
        private ConstraintLayout layout_item;
        private TextView txt_play_name;

        public ViewHolder(View view) {
            super(view);
            this.layout_item = (ConstraintLayout) view.findViewById(R.id.layout_item);
            this.img_play_icon = (ImageView) view.findViewById(R.id.img_play_icon);
            this.txt_play_name = (TextView) view.findViewById(R.id.txt_play_name);
        }
    }

    public PlaySelectorAdapter(Context context, List<PlaySelectorModel> list) {
        this.mContext = context;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        List<PlaySelectorModel> list = this.mList;
        if (list != null) {
            return list.size();
        }
        return 0;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        viewHolder.itemView.setPadding(i % 2 == 1 ? this.rowSpacingDip : 0, i >= 2 ? this.rowSpacingDip : 0, 0, 0);
        if (this.itemHeight != 0) {
            ViewGroup.LayoutParams layoutParams = viewHolder.layout_item.getLayoutParams();
            layoutParams.height = this.itemHeight;
            viewHolder.layout_item.setLayoutParams(layoutParams);
        }
        viewHolder.layout_item.setBackgroundResource(this.mList.get(i).getGradientRes());
        viewHolder.img_play_icon.setImageResource(this.mList.get(i).getIconRes());
        LanguageUtil.setText(this.mContext, viewHolder.txt_play_name, this.mList.get(i).getTitle());
        viewHolder.layout_item.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.PlaySelectorAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (PlaySelectorAdapter.this.onItemClickListener != null) {
                    PlaySelectorAdapter.this.onItemClickListener.onItemClick(i);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_play_selector, viewGroup, false));
    }

    public void setItemHeight(int i) {
        this.itemHeight = i;
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }

    public void setRowSpacingDip(int i) {
        this.rowSpacingDip = i;
    }
}
