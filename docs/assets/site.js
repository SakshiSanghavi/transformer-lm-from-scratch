(()=>{
  const $=id=>document.getElementById(id);
  document.querySelectorAll('.math[data-expr]').forEach(el=>{
    const render=()=>window.katex&&katex.render(el.dataset.expr,el,{displayMode:true,throwOnError:false});
    render(); addEventListener('load',render);
  });
  document.querySelectorAll('.copy[data-target]').forEach(button=>button.onclick=async()=>{
    try { await navigator.clipboard.writeText($(button.dataset.target).innerText); button.textContent='Copied'; setTimeout(()=>button.textContent='Copy',1100); }
    catch { button.textContent='Select text'; }
  });
  const menu=$('mobile-menu'),nav=$('site-nav');
  if(menu&&nav) menu.onclick=()=>nav.classList.toggle('open');
  if('IntersectionObserver' in window){
    const links=[...document.querySelectorAll('a[href^="#"]')];
    const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{
      if(entry.isIntersecting) links.forEach(link=>link.classList.toggle('active',link.hash==='#'+entry.target.id));
    }),{rootMargin:'-18% 0px -72% 0px'});
    document.querySelectorAll('section[id]').forEach(section=>observer.observe(section));
  }
})();
