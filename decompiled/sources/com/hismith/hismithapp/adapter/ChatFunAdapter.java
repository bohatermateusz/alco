package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.ChatFunModel;
import java.util.List;

/* loaded from: classes3.dex */
public class ChatFunAdapter extends RecyclerView.Adapter<ViewHolder> {
    public static final String TYPE_CAMERA = "fun_type_camera";
    public static final String TYPE_IMG = "fun_type_img";
    public static final String TYPE_REMOTE = "fun_type_remote";
    private Context mContext;
    private List<ChatFunModel> mList;
    private OnItemClickListener onItemClickListener;

    public interface OnItemClickListener {
        void onItemClick(String str);
    }

    protected class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView img_logo;
        private TextView txt_name;

        public ViewHolder(View view) {
            super(view);
            this.img_logo = (ImageView) view.findViewById(R.id.img_logo);
            this.txt_name = (TextView) view.findViewById(R.id.txt_name);
        }
    }

    public ChatFunAdapter(Context context, List<ChatFunModel> list) {
        this.mContext = context;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        viewHolder.txt_name.setText(this.mList.get(i).getName());
        viewHolder.img_logo.setImageResource(this.mList.get(i).getIcon());
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ChatFunAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (ChatFunAdapter.this.onItemClickListener != null) {
                    ChatFunAdapter.this.onItemClickListener.onItemClick(((ChatFunModel) ChatFunAdapter.this.mList.get(i)).getType());
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_chat_fun, viewGroup, false));
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
