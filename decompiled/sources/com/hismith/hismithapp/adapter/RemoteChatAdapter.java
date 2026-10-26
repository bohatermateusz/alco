package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.RemoteChatModel;
import java.util.List;

/* loaded from: classes3.dex */
public class RemoteChatAdapter extends RecyclerView.Adapter<ViewHolder> {
    private boolean isSingleLine = false;
    private Context mContext;
    private List<RemoteChatModel> mList;

    class ViewHolder extends RecyclerView.ViewHolder {
        private FrameLayout chat_item_layout;
        private TextView chat_item_txt_content;

        public ViewHolder(View view) {
            super(view);
            this.chat_item_layout = (FrameLayout) view.findViewById(R.id.chat_item_layout);
            this.chat_item_txt_content = (TextView) view.findViewById(R.id.chat_item_txt_content);
        }
    }

    public RemoteChatAdapter(Context context, List<RemoteChatModel> list) {
        this.mContext = context;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        List<RemoteChatModel> list = this.mList;
        if (list != null) {
            return list.size();
        }
        return 0;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemViewType(int i) {
        return this.mList.get(i).getSend() == 1 ? 1 : 0;
    }

    public Context getmContext() {
        return this.mContext;
    }

    public List<RemoteChatModel> getmList() {
        return this.mList;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, int i) {
        viewHolder.chat_item_txt_content.setSingleLine(this.isSingleLine);
        viewHolder.chat_item_txt_content.setText(this.mList.get(i).getMessage());
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return i == 1 ? new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_remote_chat_right, viewGroup, false)) : new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_remote_chat_left, viewGroup, false));
    }

    public void setSingleLine(boolean z) {
        this.isSingleLine = z;
        notifyDataSetChanged();
    }

    public void setmContext(Context context) {
        this.mContext = context;
    }

    public void setmList(List<RemoteChatModel> list) {
        this.mList = list;
    }
}
