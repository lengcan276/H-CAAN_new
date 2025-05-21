 创建新密钥(如果需要区分)
如果您想为H-CAAN仓库使用单独的密钥:
bash# 生成新的密钥对
ssh-keygen -t ed25519 -f ~/.ssh/hcaan_key -C "lengcan276lc@sina.com"

# 配置SSH使用此密钥访问H-CAAN仓库
nano ~/.ssh/config
在config文件中添加:
Host github.com-hcaan
    Hostname ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/hcaan_key
然后修改git配置:
bashgit remote set-url origin git@github.com-hcaan:lengcan276/H-CAAN.git
最后将新生成的公钥(~/.ssh/hcaan_key.pub)添加到GitHub的个人SSH密钥或部署密钥中。

在H-CAAN的deployed keys中加入hcaan_key（名称命名正确）

首次：
# 如果仓库为空，创建一个README文件
echo "# H-CAAN" > README.md
echo "层次化跨模态自适应注意力网络用于增强药物属性预测" >> README.md

# 添加文件
git add README.md

# 提交
git commit -m "Initial commit"

# 推送
git push -u origin main
