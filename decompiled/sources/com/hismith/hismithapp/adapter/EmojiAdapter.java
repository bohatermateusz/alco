package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.google.gson.Gson;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.EmojiModel;
import com.hismith.hismithapp.utils.LogUtil;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class EmojiAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Gson gson = new Gson();
    private Context mContext;
    private List<EmojiModel.Emoji> mList;
    private OnItemCLickListener onItemCLickListener;

    public interface OnItemCLickListener {
        void onItemClick(String str);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private TextView mItemEmoji;

        public ViewHolder(View view) {
            super(view);
            this.mItemEmoji = (TextView) view.findViewById(R.id.item_emoji_txt_emoji);
        }
    }

    public EmojiAdapter(Context context) {
        this.mContext = context;
        this.mList = new ArrayList();
        try {
            this.mList = new ArrayList(Arrays.asList(((EmojiModel) this.gson.fromJson(getJson("EmojiList.json", context), EmojiModel.class)).getEmoji_list()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String getJson(String str, Context context) {
        StringBuilder sb = new StringBuilder();
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(context.getAssets().open(str)));
            while (true) {
                String readLine = bufferedReader.readLine();
                if (readLine == null) {
                    break;
                }
                sb.append(readLine);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return sb.toString();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, final int i) {
        viewHolder.mItemEmoji.setText(new String(Character.toChars(this.mList.get(i).getUnicode())));
        viewHolder.mItemEmoji.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.EmojiAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (EmojiAdapter.this.onItemCLickListener != null) {
                    EmojiAdapter.this.onItemCLickListener.onItemClick(viewHolder.mItemEmoji.getText().toString());
                }
                LogUtil.e("mItemEmoji position:" + i + ", 内容：" + viewHolder.mItemEmoji.getText().toString());
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_emoji, viewGroup, false));
    }

    public void setOnItemCLickListener(OnItemCLickListener onItemCLickListener) {
        this.onItemCLickListener = onItemCLickListener;
    }
}
