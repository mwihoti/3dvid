#!/usr/bin/env bash
# Launch an EC2 instance that builds and runs the 3dvid studio. Run this in AWS CloudShell.
#   bash aws-launch.sh                          # g4dn.xlarge (T4 GPU, ~$0.53/h)
#   INSTANCE_TYPE=c6i.8xlarge bash aws-launch.sh  # CPU fallback if you have no GPU quota
set -euo pipefail
REGION=${AWS_REGION:-us-east-1}
TYPE=${INSTANCE_TYPE:-g4dn.xlarge}
NAME=3dvid-studio
MYIP=$(curl -s https://checkip.amazonaws.com)/32
TOKEN=$(openssl rand -hex 24)

echo "== key pair"
if ! aws ec2 describe-key-pairs --key-names "$NAME" --region "$REGION" >/dev/null 2>&1; then
  aws ec2 create-key-pair --key-name "$NAME" --key-type ed25519 --region "$REGION" \
    --query KeyMaterial --output text > ~/"$NAME".pem
  chmod 400 ~/"$NAME".pem
fi

echo "== security group (22 + 8080 from $MYIP only)"
VPC=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --region "$REGION" \
      --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --region "$REGION" \
     --filters Name=group-name,Values="$NAME" Name=vpc-id,Values="$VPC" \
     --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG" ] || [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --group-name "$NAME" --description "3dvid studio" \
       --vpc-id "$VPC" --region "$REGION" --query GroupId --output text)
fi
for p in 22 8080; do
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port "$p" \
    --cidr "$MYIP" --region "$REGION" >/dev/null 2>&1 || true
done

echo "== AMI (Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04)"
AMI=$(aws ssm get-parameter --region "$REGION" \
      --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
      --query Parameter.Value --output text 2>/dev/null) || \
AMI=$(aws ec2 describe-images --owners amazon --region "$REGION" \
      --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*' \
      --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
echo "   $AMI"

cat > /tmp/userdata.sh <<EOF
#!/bin/bash
set -eux
cd /home/ubuntu
git clone https://github.com/mwihoti/3dvid.git
cd 3dvid
echo "AUTH_TOKEN=$TOKEN" > .env
docker compose version >/dev/null 2>&1 || apt-get install -y docker-compose-plugin
bash deploy/ec2.sh > /var/log/3dvid-deploy.log 2>&1
chown -R ubuntu:ubuntu /home/ubuntu/3dvid
EOF

echo "== launching $TYPE"
ID=$(aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" \
     --key-name "$NAME" --security-group-ids "$SG" \
     --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":60,"VolumeType":"gp3"}}]' \
     --user-data file:///tmp/userdata.sh \
     --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" \
     --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --instance-ids "$ID" --region "$REGION"
IP=$(aws ec2 describe-instances --instance-ids "$ID" --region "$REGION" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat <<EOF

  instance : $ID
  ssh      : ssh -i ~/$NAME.pem ubuntu@$IP
  app      : http://$IP:8080      (ready in ~15 min, image build)
  token    : $TOKEN
  progress : ssh in, then  tail -f /var/log/3dvid-deploy.log
  stop     : aws ec2 stop-instances --instance-ids $ID --region $REGION
EOF
echo "$ID $IP $TOKEN" > ~/3dvid-instance.txt
