package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.ShareModel;
import java.util.List;

/* loaded from: classes3.dex */
public class ShareAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Context mContext;
    private List<ShareModel> mList;
    private OnItemCLickListener onItemCLickListener;

    public interface OnItemCLickListener {
        void onItemClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImgLogo;
        private TextView mTxtName;

        public ViewHolder(View view) {
            super(view);
            this.mImgLogo = (ImageView) view.findViewById(R.id.item_share_img_logo);
            this.mTxtName = (TextView) view.findViewById(R.id.item_share_txt_name);
        }
    }

    public ShareAdapter(Context context, List<ShareModel> list) {
        this.mContext = context;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        viewHolder.mImgLogo.setImageDrawable(this.mContext.getResources().getDrawable(this.mList.get(i).getLogoInt()));
        viewHolder.mTxtName.setText(this.mList.get(i).getName());
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ShareAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (ShareAdapter.this.onItemCLickListener != null) {
                    ShareAdapter.this.onItemCLickListener.onItemClick(((ShareModel) ShareAdapter.this.mList.get(i)).getShareId());
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_share, viewGroup, false));
    }

    public void setOnItemCLickListener(OnItemCLickListener onItemCLickListener) {
        this.onItemCLickListener = onItemCLickListener;
    }
}
