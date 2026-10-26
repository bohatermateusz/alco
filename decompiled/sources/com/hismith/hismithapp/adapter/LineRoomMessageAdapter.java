package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.text.SpannableString;
import android.text.TextPaint;
import android.text.TextUtils;
import android.text.method.LinkMovementMethod;
import android.text.style.ClickableSpan;
import android.text.style.ForegroundColorSpan;
import android.text.style.UnderlineSpan;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.socket.PlayLineSocketModel;
import com.hismith.hismithapp.utils.IntentUtil;
import java.util.List;

/* loaded from: classes3.dex */
public class LineRoomMessageAdapter extends RecyclerView.Adapter<ViewHolder> {
    private boolean isSingleLine = false;
    private Context mContext;
    private List<PlayLineSocketModel> mMessages;

    class NoUnderlineSpan extends UnderlineSpan {
        NoUnderlineSpan() {
        }

        @Override // android.text.style.UnderlineSpan, android.text.style.CharacterStyle
        public void updateDrawState(TextPaint textPaint) {
            textPaint.setColor(textPaint.linkColor);
            textPaint.setUnderlineText(false);
        }
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private TextView mTxtMessage;

        public ViewHolder(View view) {
            super(view);
            this.mTxtMessage = (TextView) view.findViewById(R.id.item_line_room_txt_message);
        }
    }

    public LineRoomMessageAdapter(Context context, List<PlayLineSocketModel> list) {
        this.mContext = context;
        this.mMessages = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mMessages.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        ClickableSpan clickableSpan = new ClickableSpan() { // from class: com.hismith.hismithapp.adapter.LineRoomMessageAdapter.1
            @Override // android.text.style.ClickableSpan
            public void onClick(View view) {
                try {
                    IntentUtil.startUserInfoActivity(LineRoomMessageAdapter.this.mContext, Integer.valueOf(((PlayLineSocketModel) LineRoomMessageAdapter.this.mMessages.get(i)).getUserId()).intValue());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        };
        viewHolder.mTxtMessage.setSingleLine(this.isSingleLine);
        if (!"*#*".equals(this.mMessages.get(i).getUserName())) {
            SpannableString spannableString = new SpannableString(this.mMessages.get(i).getUserName() + ": " + this.mMessages.get(i).getMessage());
            if (!TextUtils.isEmpty(this.mMessages.get(i).getUserName())) {
                spannableString.setSpan(clickableSpan, 0, this.mMessages.get(i).getUserName().length(), 33);
                spannableString.setSpan(new NoUnderlineSpan(), 0, this.mMessages.get(i).getUserName().length(), 17);
            }
            spannableString.setSpan(new ForegroundColorSpan(this.mContext.getResources().getColor(R.color.app_style_color)), 0, this.mMessages.get(i).getUserName().length() + 2, 33);
            viewHolder.mTxtMessage.setMovementMethod(LinkMovementMethod.getInstance());
            viewHolder.mTxtMessage.setText(spannableString);
            return;
        }
        String message = this.mMessages.get(i).getMessage();
        int indexOf = message.indexOf("[");
        String replace = message.replace("[", "");
        int indexOf2 = replace.indexOf("]");
        SpannableString spannableString2 = new SpannableString(replace.replace("]", ""));
        spannableString2.setSpan(clickableSpan, indexOf, indexOf2, 33);
        spannableString2.setSpan(new NoUnderlineSpan(), indexOf, indexOf2, 17);
        spannableString2.setSpan(new ForegroundColorSpan(this.mContext.getResources().getColor(R.color.app_style_color)), indexOf, indexOf2, 33);
        viewHolder.mTxtMessage.setMovementMethod(LinkMovementMethod.getInstance());
        viewHolder.mTxtMessage.setText(spannableString2);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_line_room_message, viewGroup, false));
    }

    public void setSingleLine(boolean z) {
        this.isSingleLine = z;
        notifyDataSetChanged();
    }
}
