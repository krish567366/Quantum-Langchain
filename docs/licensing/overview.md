# 🔐 QuantumLangChain Licensing Overview

## Important Notice

**QuantumLangChain is commercial software requiring a valid license for all features beyond the 24-hour evaluation period.**

---

## 📋 Table of Contents

1. [License Tiers](#license-tiers)
2. [Getting Started](#getting-started)
3. [Evaluation Period](#evaluation-period)
4. [Feature Matrix](#feature-matrix)
5. [Usage Limits](#usage-limits)
6. [Contact Information](#contact-information)
7. [FAQ](#faq)

---

## License Tiers

### 🆓 Evaluation Tier
- **Duration**: 24 hours from first use
- **Price**: Free
- **Features**: Basic features only
- **Operations**: 1,000 per day
- **Purpose**: Evaluation and testing

### 💼 Basic Tier
- **Price**: $29/month
- **Features**: Core functionality
  - QLChain basic operations
  - Quantum Memory (basic)
  - Simple backends
  - Basic chains
- **Operations**: 10,000 per day
- **Support**: Email support

### 🚀 Professional Tier
- **Price**: $99/month
- **Features**: Advanced functionality
  - All Basic features
  - Multi-Agent Systems
  - EntangledAgents
  - Advanced backends
  - Quantum Retrieval
  - Quantum Tools
- **Operations**: 100,000 per day
- **Support**: Priority email support

### 🏢 Enterprise Tier
- **Price**: $299/month
- **Features**: Complete functionality
  - All Professional features
  - Distributed systems
  - Custom backends
  - Advanced analytics
  - Enterprise integrations
- **Operations**: Unlimited
- **Support**: Priority support + Phone

### 🎓 Research Tier
- **Price**: $49/month (Academic only)
- **Features**: Research-specific tools
  - Core features
  - Experimental APIs
  - Research backends
  - Academic license terms
- **Operations**: 50,000 per day
- **Support**: Academic support

---

## Getting Started

### Step 1: Installation
```bash
pip install quantumlangchain
```

### Step 2: First Import
```python
import quantumlangchain as qlc
# 24-hour evaluation period starts automatically
```

### Step 3: Get Machine ID
```python
machine_id = qlc.get_machine_id()
print(f"Your Machine ID: {machine_id}")
```

### Step 4: Contact for License
- **Email**: bajpaikrishna715@gmail.com
- **Include**: Your machine ID
- **Specify**: Desired license tier
- **Response**: Within 24 hours

### Step 5: Activate License
```python
# Once you receive your license file
from quantumlangchain import LicenseManager
license_manager = LicenseManager()
license_manager.activate_license("path/to/license.qkey")
```

---

## Evaluation Period

### What's Included
- **Duration**: 24 hours from first import
- **Features**: Limited to evaluation tier
- **Operations**: Up to 1,000 operations
- **Purpose**: Testing and evaluation

### What Happens After
- **Grace Period Expires**: All features become inaccessible
- **Contact Required**: Email bajpaikrishna715@gmail.com
- **Purchase License**: Choose appropriate tier
- **Immediate Access**: Resume work after activation

### Checking Status
```python
import quantumlangchain as qlc

# Display comprehensive license information
qlc.display_license_info()

# Check specific status
status = qlc.get_license_status()
if status['grace_active']:
    hours_remaining = status['grace_remaining_hours']
    print(f"Grace period: {hours_remaining:.1f} hours remaining")
```

---

## Feature Matrix

| Feature | Evaluation | Basic | Professional | Enterprise | Research |
|---------|------------|-------|--------------|------------|----------|
| **Core Features** |
| QLChain Basic | ✅ | ✅ | ✅ | ✅ | ✅ |
| Quantum Memory Basic | ✅ | ✅ | ✅ | ✅ | ✅ |
| Simple Backends | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Professional Features** |
| Multi-Agent Systems | ❌ | ❌ | ✅ | ✅ | ❌ |
| EntangledAgents | ❌ | ❌ | ✅ | ✅ | ❌ |
| Advanced Backends | ❌ | ❌ | ✅ | ✅ | ❌ |
| Quantum Retrieval | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Enterprise Features** |
| Distributed Systems | ❌ | ❌ | ❌ | ✅ | ❌ |
| Custom Backends | ❌ | ❌ | ❌ | ✅ | ❌ |
| Advanced Analytics | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Research Features** |
| Experimental APIs | ❌ | ❌ | ❌ | ❌ | ✅ |
| Research Backends | ❌ | ❌ | ❌ | ❌ | ✅ |
| Academic Terms | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## Usage Limits

### Daily Operation Limits

| Tier | Operations/Day | Overage Policy |
|------|----------------|----------------|
| Evaluation | 1,000 | Hard limit |
| Basic | 10,000 | Contact for overage |
| Professional | 100,000 | Contact for overage |
| Enterprise | Unlimited | N/A |
| Research | 50,000 | Contact for overage |

### Monitoring Usage
```python
import quantumlangchain as qlc

status = qlc.get_license_status()
print(f"Today's usage: {status['usage_today']} operations")

# Usage resets daily at midnight UTC
```

### What Counts as an Operation
- QLChain execution (`chain.arun()`)
- Memory storage/retrieval
- Agent collaboration tasks
- Quantum backend calls
- Retrieval operations

---

## Contact Information

### Primary Contact
- **Email**: bajpaikrishna715@gmail.com
- **Response Time**: Within 24 hours
- **Include**: Always include your machine ID

### What to Include
```python
import quantumlangchain as qlc
machine_id = qlc.get_machine_id()
status = qlc.get_license_status()

print(f"Machine ID: {machine_id}")
print(f"Current Status: {status}")
```

### Support Types

#### Sales Inquiries
- License tier recommendations
- Pricing questions
- Custom requirements
- Academic discounts

#### Technical Support
- License activation issues
- Feature questions
- Integration help
- Bug reports

#### Billing Support
- Payment questions
- Invoice requests
- License renewals
- Upgrades/downgrades

---

## CLI Tools

### License Management CLI
```bash
# Check license status
quantum-license status

# Get machine ID
quantum-license machine-id

# Test component access
quantum-license test

# Get support information
quantum-license support

# Show license information
quantum-license info
```

---

## FAQ

### General Questions

**Q: Why is QuantumLangChain licensed software?**
A: QuantumLangChain represents significant research and development investment in cutting-edge quantum-classical AI technology. Licensing ensures sustainable development and high-quality support.

**Q: Can I use QuantumLangChain for open source projects?**
A: Yes, with an appropriate license. Contact us to discuss your open source project requirements.

**Q: Is there a student discount?**
A: Yes, the Research tier ($49/month) is available for academic use with proper verification.

### Technical Questions

**Q: What happens if I exceed my operation limit?**
A: Operations will be blocked until the next day (midnight UTC) or you upgrade your tier.

**Q: Can I upgrade my license tier anytime?**
A: Yes, contact bajpaikrishna715@gmail.com for immediate upgrades.

**Q: Do I need internet connection for license validation?**
A: Initial validation requires internet, but the software can work offline for short periods.

### Billing Questions

**Q: Are licenses per-user or per-machine?**
A: Licenses are per-machine based on hardware fingerprinting.

**Q: Can I transfer my license to a new machine?**
A: Yes, contact support for license transfers.

**Q: What payment methods do you accept?**
A: We accept credit cards, bank transfers, and purchase orders for enterprises.

### Development Questions

**Q: Can I develop and test without a license?**
A: Yes, use the 24-hour evaluation period or set `QUANTUMLANGCHAIN_DEV=1` for development mode.

**Q: How do I handle licensing in CI/CD?**
A: Set the development environment variable or contact us for CI/CD licensing options.

**Q: Can I mock license validation in tests?**
A: Yes, use the provided test fixtures to mock license validation.

---

## Next Steps

1. **Try the Evaluation**: Install and explore for 24 hours
2. **Choose Your Tier**: Based on your feature requirements
3. **Contact Us**: Email bajpaikrishna715@gmail.com with your machine ID
4. **Get Licensed**: Receive and activate your license
5. **Start Building**: Create amazing quantum-classical AI applications

---

**📧 Contact: bajpaikrishna715@gmail.com**  
**🔧 Always include your machine ID when contacting support**  
**⚡ Response within 24 hours**
